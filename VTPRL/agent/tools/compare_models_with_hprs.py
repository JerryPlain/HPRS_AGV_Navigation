import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

# Add parent directory (agent/) to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import Config
from envs.get_env import get_env
from expert.expert_dataset import ExpertDataset
from td3.td3bc.td3_bc_agent import TD3BCAgent
from td3.online_td3.egocentric_normalization_wrapper import EgocentricNormalizationWrapper


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _evaluate_model(
    env,
    agent: TD3BCAgent,
    num_episodes: int,
    max_steps: int,
    randomize_seed: bool,
    base_seed: int,
    seed_stride: int,
    verbose: bool,
) -> Dict[str, float]:
    successes: List[float] = []
    collisions: List[float] = []
    timeouts: List[float] = []
    rewards: List[float] = []
    success_rewards: List[float] = []
    steps_all: List[int] = []
    steps_success: List[int] = []

    for ep in range(num_episodes):
        if randomize_seed:
            env.seed(base_seed + ep * seed_stride)

        obs = env.reset()
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            obs = obs[0]

        done = False
        total_reward = 0.0
        steps = 0
        info_dict: Dict[str, Any] = {}

        while not done and steps < max_steps:
            state = obs.astype(np.float32)
            action = agent.select_action(state, noise=0.0)
            next_obs, reward, dones, infos = env.step([action])
            obs = next_obs[0]
            total_reward += float(reward[0])
            done = bool(dones[0])
            info_dict = infos[0] if isinstance(infos, (list, tuple)) else infos
            steps += 1

        success = bool(info_dict.get("success", False)) or bool(info_dict.get("is_success", False))
        collision = bool(info_dict.get("collision", False)) or bool(info_dict.get("hprs/collision", False))
        timeout = bool(info_dict.get("timeout", False))

        rewards.append(total_reward)
        steps_all.append(steps)
        successes.append(1.0 if success else 0.0)
        collisions.append(1.0 if collision else 0.0)
        timeouts.append(1.0 if timeout else 0.0)
        if success:
            success_rewards.append(total_reward)
            steps_success.append(steps)

        if verbose:
            print(
                f"[Eval] ep={ep+1}/{num_episodes} "
                f"success={int(success)} collision={int(collision)}"
            )

    return {
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "timeout_rate": float(np.mean(timeouts)),
        "mean_reward": float(np.mean(rewards)),
        "mean_reward_success": float(np.mean(success_rewards)) if success_rewards else float("nan"),
        "mean_steps": float(np.mean(steps_all)),
        "mean_steps_success": float(np.mean(steps_success)) if steps_success else float("nan"),
    }


def _try_plot(results: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Compare] matplotlib not found. Skipping plot.")
        return

    labels = list(results.keys())
    success = [results[k]["success_rate"] for k in labels]
    collision = [results[k]["collision_rate"] for k in labels]
    success_rew = [results[k]["mean_reward_success"] for k in labels]
    steps = [results[k]["mean_steps"] for k in labels]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    bars0 = axs[0].bar(labels, success)
    axs[0].set_title("Success Rate")
    axs[0].set_ylim(0, 1.0)

    bars1 = axs[1].bar(labels, collision)
    axs[1].set_title("Collision Rate")
    axs[1].set_ylim(0, 1.0)

    bars2 = axs[2].bar(labels, success_rew)
    axs[2].set_title("Mean Success Reward")

    bars3 = axs[3].bar(labels, steps)
    axs[3].set_title("Mean Steps")

    def _annotate(ax, bars, fmt="{:.2f}"):
        for b in bars:
            val = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                val,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    _annotate(axs[0], bars0, fmt="{:.2f}")
    _annotate(axs[1], bars1, fmt="{:.2f}")
    _annotate(axs[2], bars2, fmt="{:.2f}")
    _annotate(axs[3], bars3, fmt="{:.1f}")

    for ax in axs:
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    out_path = out_dir / "compare_models_bars.png"
    fig.savefig(out_path, dpi=150)
    print(f"[Compare] Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple models with per-model HPRS configs.")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluate_compare_models.yaml")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode results")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    dataset_path = cfg["dataset_path"]
    num_episodes = int(cfg.get("num_episodes", 50))
    max_steps = int(cfg.get("max_steps", 2000))
    randomize_seed = bool(cfg.get("randomize_seed", False))
    base_seed = int(cfg.get("base_seed", 0))
    seed_stride = int(cfg.get("seed_stride", 1))
    out_dir = Path(cfg.get("out_dir", "./agent/logs/compare_models"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = cfg.get("cases", [])
    if not cases:
        raise ValueError("No cases found in config.")

    main_config = Config()
    agent_cfg = main_config.agent_dict
    root = main_config.root_dict
    sim = main_config.simulation_dict
    gym = main_config.gym_environment_dict
    manip_env = main_config.manipulator_environment_dict
    warehouse_env = main_config.warehouse_environment_dict
    observation = main_config.observation_dict
    reward_dict = gym["manipulator_gym_environment"]["reward"]

    print(f"[Compare] Loading dataset stats: {dataset_path}")
    dataset = ExpertDataset(path=dataset_path)
    stats = dataset.stats

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict[str, Dict[str, float]] = {}

    for case in cases:
        name = case["name"]
        model_prefix = case["model_prefix"]
        hprs_config = case["hprs_config"]

        print(f"\n[Compare] Case: {name}")
        print(f"[Compare]   model: {model_prefix}")
        print(f"[Compare]   hprs:  {hprs_config}")

        gym_case = dict(gym)
        gym_case["num_envs"] = 1
        agent_cfg_case = dict(agent_cfg)
        agent_cfg_case["log_dir"] = str(out_dir / name)

        env = get_env(
            agent_cfg_case, root, sim, gym_case, manip_env, warehouse_env,
            observation, reward_dict, agent_cfg_case["log_dir"],
            hprs_config,
            additional_wrapper=EgocentricNormalizationWrapper,
            wrapper_kwargs={
                "stats": stats,
                "normalize_obs": True,
                "unnormalize_actions": True,
            },
        )

        agent = TD3BCAgent(
            state_dim=stats["state_dim"],
            action_dim=stats["action_dim"],
            max_action=1.0,
            device=device,
        )
        agent.load(model_prefix)
        agent.actor.to(device)
        agent.critic.to(device)
        agent.device = device

        metrics = _evaluate_model(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            max_steps=max_steps,
            randomize_seed=randomize_seed,
            base_seed=base_seed,
            seed_stride=seed_stride,
            verbose=args.verbose,
        )
        results[name] = metrics
        print(
            f"[{name}] success={metrics['success_rate']:.1%}, "
            f"collision={metrics['collision_rate']:.1%}, "
            f"mean_reward={metrics['mean_reward']:.2f}"
        )

        env.close()

    csv_path = out_dir / "compare_models.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "name",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "mean_reward",
            "mean_reward_success",
            "mean_steps",
            "mean_steps_success",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, metrics in results.items():
            row = {"name": name}
            row.update(metrics)
            w.writerow(row)

    print(f"[Compare] Saved CSV: {csv_path}")
    _try_plot(results, out_dir)


if __name__ == "__main__":
    main()
