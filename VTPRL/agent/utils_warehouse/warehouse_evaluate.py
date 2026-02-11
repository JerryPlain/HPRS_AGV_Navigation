import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]   # utils_warehouse -> agent -> VTPRL
testcases_yaml = REPO_ROOT / "testcases" / "warehouse_test_cases.yaml"
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO

def load_testcases(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    cases = data.get("cases", []) if isinstance(data, dict) else []
    return meta, cases

def _extract_flag(info, env_i: int, key: str, default=False) -> bool:
    # info can be list[dict] or dict
    if isinstance(info, (list, tuple)):
        if env_i < len(info) and isinstance(info[env_i], dict):
            return bool(info[env_i].get(key, default))
        return bool(default)
    if isinstance(info, dict):
        return bool(info.get(key, default))
    return bool(default)

def rollout_on_testcases(env, model, cases, repeats=1, deterministic=True, tb_writer=None, global_step=0):
    """
    env: SB3 VecEnv
    model: PPO model (has predict)
    cases: list of dict testcases
    """
    returns = []
    lengths = []
    succ = []
    coll = []
    tout = []

    num_envs = getattr(env, "num_envs", None)
    if num_envs is None:
        # fallback: some wrappers store it differently
        num_envs = getattr(env, "n_envs", 1)

    # evaluation mode on env side (optional)
    if hasattr(env, "env_method"):
        try:
            env.env_method("set_evaluation_mode", True)
        except Exception:
            pass

    for rep in range(int(repeats)):
        for tc in cases:
            # broadcast testcase to all sub-envs
            if hasattr(env, "env_method"):
                env.env_method("set_test_case", tc)

            obs = env.reset()
            ep_return = np.zeros(num_envs, dtype=np.float64)
            ep_len = np.zeros(num_envs, dtype=np.int32)

            while True:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, info = env.step(action)

                ep_return += np.asarray(rewards).reshape(-1)
                ep_len += 1

                if np.any(dones):
                    for i in range(num_envs):
                        returns.append(float(ep_return[i]))
                        lengths.append(int(ep_len[i]))
                        succ.append(1 if _extract_flag(info, i, "success", False) else 0)
                        coll.append(1 if _extract_flag(info, i, "collision", False) else 0)
                        tout.append(1 if _extract_flag(info, i, "timeout", False) else 0)
                    break

    mean_return = float(np.mean(returns)) if returns else 0.0
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    success_rate = float(np.mean(succ)) if succ else 0.0
    collision_rate = float(np.mean(coll)) if coll else 0.0
    timeout_rate = float(np.mean(tout)) if tout else 0.0

    if tb_writer is not None:
        tb_writer.add_scalar("eval/mean_return", mean_return, global_step)
        tb_writer.add_scalar("eval/mean_ep_len", mean_len, global_step)
        tb_writer.add_scalar("eval/success_rate", success_rate, global_step)
        tb_writer.add_scalar("eval/collision_rate", collision_rate, global_step)
        tb_writer.add_scalar("eval/timeout_rate", timeout_rate, global_step)
        tb_writer.flush()

    metrics = {
        "mean_return": mean_return,
        "mean_ep_len": mean_len,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "n_episodes": len(returns),
    }
    return metrics

def evaluate_single_checkpoint(env, agent, checkpoint_path: str, testcases_yaml: str):
    meta, cases = load_testcases(testcases_yaml)
    repeats = meta.get("repeats", 1)

    model = PPO.load(checkpoint_path)
    model.policy.set_training_mode(False)

    log_dir = agent["log_dir"]
    Path(os.path.join(log_dir, "evaluation_single_dfs")).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir, "evaluation_tb_single"))

    metrics = rollout_on_testcases(env, model, cases, repeats=repeats, deterministic=True, tb_writer=writer, global_step=0)
    writer.close()

    df = pd.DataFrame([{
        "evaluation_name": agent.get("evaluation_name", "warehouse_eval"),
        "checkpoint": checkpoint_path,
        **metrics
    }])
    out_csv = os.path.join(log_dir, "evaluation_single_dfs", "df_evaluation_single.csv")
    df.to_csv(out_csv, index=False)
    return metrics, out_csv
