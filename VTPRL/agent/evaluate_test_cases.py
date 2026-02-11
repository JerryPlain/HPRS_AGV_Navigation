import os
import csv
import yaml
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

from utils_warehouse.warehouse_evaluate import evaluate_single_checkpoint


def _load_cases(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    defaults = data.get("defaults", {}) or {}
    cases = data.get("cases", []) or []
    return defaults, cases


def _safe_env_method(env, name: str, *args, **kwargs) -> bool:
    """Call env.env_method if VecEnv, else call directly if present."""
    try:
        env.env_method(name, *args, **kwargs)
        return True
    except Exception:
        if hasattr(env, name):
            getattr(env, name)(*args, **kwargs)
            return True
    return False


def _get_max_speeds(env) -> Tuple[float, float]:
    """Fetch max_linear_velocity/max_angular_velocity from VecEnv if possible; else fallback to action_space.high."""
    try:
        max_v = float(env.get_attr("max_linear_velocity")[0])
        max_w = float(env.get_attr("max_angular_velocity")[0])
        return max_v, max_w
    except Exception:
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        return float(high[0]), float(high[1])


def _make_out_csv_path(agent_cfg: Dict[str, Any], algo: str) -> str:
    log_dir = agent_cfg.get("log_dir", ".")
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"warehouse_eval_{algo}_{stamp}.csv")


def _compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"num_cases": 0, "success_rate": 0.0, "collision_rate": 0.0, "timeout_rate": 0.0}

    succ = sum(1 for r in rows if r.get("success"))
    col = sum(1 for r in rows if r.get("collision"))
    tout = sum(1 for r in rows if r.get("timeout"))

    avg_steps = float(np.mean([r.get("steps", 0) for r in rows]))
    avg_return = float(np.mean([r.get("episode_return", 0.0) for r in rows]))

    return {
        "num_cases": n,
        "success": succ,
        "collision": col,
        "timeout": tout,
        "success_rate": succ / n,
        "collision_rate": col / n,
        "timeout_rate": tout / n,
        "avg_steps": avg_steps,
        "avg_return": avg_return,
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "case_id",
        "steps",
        "episode_return",
        "success",
        "collision",
        "timeout",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _eval_td3bc_testcases(
    env,
    agent_cfg: Dict[str, Any],
    testcases_yaml: str,
    model_prefix: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Evaluate a TD3-BC model (prefix + _actor/_critic pth) on warehouse testcases.

    Assumptions:
    - env is a VecEnv-like wrapper (gym num_envs >= 1)
    - env.reset() returns obs shaped (num_envs, obs_dim) or (obs_dim,) for single env
    - env.step(action_batch) returns (obs, rewards, dones, infos)
    - env implements set_evaluation_mode(True) and set_test_case(tc) (via env_method)
    - TD3Agent.select_action expects a single state (obs_dim,) and outputs (2,) in [-1,1]
    """
    from td3.td3bc.td3_bc_agent import TD3BCAgent

    # turn on strict evaluation termination inside env
    _safe_env_method(env, "set_evaluation_mode", True)

    defaults, cases = _load_cases(testcases_yaml)
    if not cases:
        raise RuntimeError(f"No test cases found in: {testcases_yaml}")

    # infer dims
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    td3 = TD3BCAgent(obs_dim, act_dim, max_action=1.0, device=device)
    td3.load(model_prefix)
    # load() uses map_location=cpu, so move nets to device and set td3.device
    td3.actor.to(device)
    td3.critic.to(device)
    td3.device = device

    max_v, max_w = _get_max_speeds(env)

    # rollout params
    max_steps = int(agent_cfg.get("max_time_steps", agent_cfg.get("max_steps", 2000)))

    out_csv = _make_out_csv_path(agent_cfg, algo="td3bc")
    rows: List[Dict[str, Any]] = []

    for tc in cases:
        case_id = tc.get("id", "unknown")
        merged = dict(defaults)
        merged.update(tc)

        ok = _safe_env_method(env, "set_test_case", merged)
        if not ok:
            raise RuntimeError("Env does not support set_test_case (check WarehouseUnityEnv / wrapper).")

        obs = env.reset()

        # normalize obs to (num_envs, obs_dim)
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        done_any = False
        steps = 0
        ep_ret = 0.0
        last_info: Any = None

        while (not done_any) and steps < max_steps:
            # compute action per env
            actions = []
            for i in range(obs.shape[0]):
                a_norm = td3.select_action(obs[i])  # (2,) in [-1,1]
                a = np.asarray(a_norm, dtype=np.float32).reshape(-1)
                a_phys = np.array([a[0] * max_v, a[1] * max_w], dtype=np.float32)
                a_phys = np.clip(a_phys, env.action_space.low, env.action_space.high)
                actions.append(a_phys)

            action_batch = np.stack(actions, axis=0)

            obs, rewards, dones, infos = env.step(action_batch)

            # update accumulators
            rewards_np = np.asarray(rewards, dtype=np.float32).reshape(-1)
            ep_ret += float(np.sum(rewards_np))  # sum over envs
            steps += 1

            obs = np.asarray(obs, dtype=np.float32)
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)

            dones_np = np.asarray(dones).reshape(-1)
            done_any = bool(np.any(dones_np))
            last_info = infos

        # Extract info[0] for single-env semantics
        info0 = None
        try:
            if isinstance(last_info, (list, tuple)) and len(last_info) > 0:
                info0 = last_info[0]
            elif isinstance(last_info, dict):
                info0 = last_info
        except Exception:
            info0 = None

        success = bool(info0.get("success", False)) if isinstance(info0, dict) else False
        collision = bool(info0.get("collision", False)) if isinstance(info0, dict) else False
        timeout = bool(info0.get("timeout", False)) if isinstance(info0, dict) else False

        rows.append(
            {
                "case_id": case_id,
                "steps": steps,
                "episode_return": ep_ret,
                "success": success,
                "collision": collision,
                "timeout": timeout,
            }
        )

    _write_csv(out_csv, rows)
    metrics = _compute_metrics(rows)
    return metrics, out_csv


def run_warehouse_testcase_eval(
    env,
    agent_cfg,
    testcases_yaml,
    algo: str = "ppo",
    checkpoint_path: Optional[str] = None,
    model_prefix: Optional[str] = None,
):
    """
    Unified entrypoint used by main.py

    - algo="ppo": use existing evaluate_single_checkpoint (SB3 PPO)
    - algo="td3bc": run TD3-BC testcase rollout here (prefix-based pth weights)
    """
    algo = (algo or "ppo").lower()

    # Always try to enable strict env evaluation mode
    _safe_env_method(env, "set_evaluation_mode", True)

    if algo == "ppo":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for algo='ppo'")
        metrics, out_csv = evaluate_single_checkpoint(
            env=env,
            agent=agent_cfg,
            checkpoint_path=checkpoint_path,
            testcases_yaml=testcases_yaml,
        )
        return metrics, out_csv

    if algo in ("td3bc", "td3-bc", "td3_bc"):
        if model_prefix is None:
            raise ValueError("model_prefix is required for algo='td3bc' (prefix without _actor.pth)")
        return _eval_td3bc_testcases(
            env=env,
            agent_cfg=agent_cfg,
            testcases_yaml=testcases_yaml,
            model_prefix=model_prefix,
        )

    raise ValueError(f"Unknown algo: {algo}")
