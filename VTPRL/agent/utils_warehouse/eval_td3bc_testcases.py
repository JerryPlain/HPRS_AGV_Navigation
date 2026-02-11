import os, sys, argparse, yaml
import numpy as np
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)

from config import Config
from main import get_env
from agent.td3.td3bc.td3_bc_agent import TD3BCAgent


def load_cases(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    defaults = data.get("defaults", {})
    cases = data.get("cases", [])
    return defaults, cases


def try_env_method(env, name, *args, **kwargs):
    try:
        env.env_method(name, *args, **kwargs)
        return True
    except Exception:
        if hasattr(env, name):
            getattr(env, name)(*args, **kwargs)
            return True
    return False


def get_max_speeds(env):
    try:
        max_v = float(env.get_attr("max_linear_velocity")[0])
        max_w = float(env.get_attr("max_angular_velocity")[0])
        return max_v, max_w
    except Exception:
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        return float(high[0]), float(high[1])


def main(model_prefix: str, cases_yaml: str, max_steps: int):
    # Build env
    cfg = Config()
    agent_cfg = cfg.agent_dict
    root = cfg.root_dict
    sim = cfg.simulation_dict
    gym = cfg.gym_environment_dict
    manip_env = cfg.manipulator_environment_dict
    warehouse_env = cfg.warehouse_environment_dict
    obs_cfg = cfg.observation_dict
    reward_dict = gym["manipulator_gym_environment"]["reward"]

    env = get_env(agent_cfg, root, sim, gym, manip_env, warehouse_env, obs_cfg, reward_dict, agent_cfg["log_dir"])

    # Turn on evaluation behavior inside env (collision/timeout done + testcase enabled)
    try_env_method(env, "set_evaluation_mode", True)

    # Load testcases
    defaults, cases = load_cases(cases_yaml)
    if not cases:
        raise RuntimeError(f"No cases found in {cases_yaml}")

    # Build TD3BC agent
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    td3 = TD3BCAgent(obs_dim, act_dim, max_action=1.0, device=device)
    td3.load(model_prefix)              # expects prefix; loads *_actor.pth/_critic.pth
    td3.actor.to(device)                # load() maps to cpu, so move to device
    td3.critic.to(device)
    td3.device = device

    max_v, max_w = get_max_speeds(env)

    print("======================================")
    print("[EVAL TD3-BC TESTCASES]")
    print(" model_prefix:", model_prefix)
    print(" cases_yaml  :", cases_yaml)
    print(" device      :", device)
    print(" obs_dim/act_dim:", obs_dim, act_dim)
    print(" max_v/max_w :", max_v, max_w)
    print("======================================")

    results = []

    for tc in cases:
        tc_id = tc.get("id", "unknown")
        merged = dict(defaults)
        merged.update(tc)

        # set testcase
        ok = try_env_method(env, "set_test_case", merged)
        if not ok:
            raise RuntimeError("Env has no set_test_case; check warehouse_unity_env.py / wrapper")

        obs = env.reset()
        done = False
        steps = 0
        ep_ret = 0.0
        last_info = {}

        while (not done) and steps < max_steps:
            o = np.asarray(obs, dtype=np.float32).reshape(-1)
            a_norm = td3.select_action(o)  # [-1,1]
            a_norm = np.asarray(a_norm, dtype=np.float32).reshape(-1)

            # map to physical action
            a_phys = np.array([a_norm[0] * max_v, a_norm[1] * max_w], dtype=np.float32)
            a_phys = np.clip(a_phys, env.action_space.low, env.action_space.high)

            obs, reward, done, info = env.step(a_phys)
            ep_ret += float(reward)
            steps += 1
            last_info = info

        row = {
            "id": tc_id,
            "steps": steps,
            "return": ep_ret,
            "success": bool(last_info.get("success", False)),
            "collision": bool(last_info.get("collision", False)),
            "timeout": bool(last_info.get("timeout", False)),
        }
        results.append(row)
        print("[CASE]", row)

    # Summary
    succ = sum(r["success"] for r in results)
    col = sum(r["collision"] for r in results)
    tout = sum(r["timeout"] for r in results)
    print("======================================")
    print(f"[SUMMARY] n={len(results)} success={succ} collision={col} timeout={tout}")
    print("======================================")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="prefix path, e.g. .../td3_bc_offline_best (no _actor.pth)")
    p.add_argument("--cases", required=True, help="yaml path for testcases")
    p.add_argument("--max_steps", type=int, default=2000)
    args = p.parse_args()
    main(args.model, args.cases, args.max_steps)
