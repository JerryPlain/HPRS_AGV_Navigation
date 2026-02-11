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
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _read_csv(path: str) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("checkpoint", "").strip()
            if not label:
                continue
            metrics: Dict[str, float] = {}
            for key, val in row.items():
                if key == "checkpoint":
                    continue
                try:
                    metrics[key] = float(val)
                except Exception:
                    metrics[key] = float("nan")
            results[label] = metrics
    return results


def _evaluate_model(
    env,
    agent: TD3BCAgent,
    num_episodes: int,
    max_steps: int,
    randomize_seed: bool,
    base_seed: int,
    seed_stride: int,
    verbose: bool = False,
) -> Dict[str, float]:
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    successes: List[float] = []
    collisions: List[float] = []
    timeouts: List[float] = []
    success_rewards: List[float] = []
    success_steps: List[int] = []

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

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(1.0 if success else 0.0)
        collisions.append(1.0 if collision else 0.0)
        timeouts.append(1.0 if timeout else 0.0)
        if success:
            success_rewards.append(total_reward)
            success_steps.append(steps)

        if verbose:
            print(
                f"[Eval] ep={ep+1}/{num_episodes} "
                f"success={int(success)} collision={int(collision)}"
            )

    return {
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "timeout_rate": float(np.mean(timeouts)),
        "mean_reward_success": float(np.mean(success_rewards)) if success_rewards else float("nan"),
        "std_reward_success": float(np.std(success_rewards)) if success_rewards else float("nan"),
        "mean_steps": float(np.mean(episode_lengths)),
        "mean_steps_success": float(np.mean(success_steps)) if success_steps else float("nan"),
    }


def _try_plot(results: Dict[str, Dict[str, float]], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Compare] matplotlib not found. Writing CSV only.")
        return

    labels = list(results.keys())
    x = np.arange(len(labels))
    pretty_labels = [lbl.replace("td3_bc_offline_", "").replace("step_", "") for lbl in labels]
    success = [results[k]["success_rate"] for k in labels]
    reward = [results[k]["mean_reward_success"] for k in labels]
    steps = [results[k]["mean_steps_success"] for k in labels]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    bars0 = axs[0].bar(x, success)
    axs[0].set_title("Success Rate")
    axs[0].set_ylim(0, 1.0)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(pretty_labels, rotation=25, ha="right")
    for b in bars0:
        h = b.get_height()
        axs[0].annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=8)

    bars1 = axs[1].bar(x, reward)
    axs[1].set_title("Mean Reward (Success Only)")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(pretty_labels, rotation=25, ha="right")
    for b in bars1:
        h = b.get_height()
        axs[1].annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=8)

    bars2 = axs[2].bar(x, steps)
    axs[2].set_title("Mean Steps (Success Only)")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(pretty_labels, rotation=25, ha="right")
    for b in bars2:
        h = b.get_height()
        axs[2].annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "compare_checkpoints_bars.png")
    fig.savefig(out_path, dpi=150)
    print(f"[Compare] Saved bar plot: {out_path}")

    fig2, axs2 = plt.subplots(1, 3, figsize=(12, 4))
    axs2[0].plot(x, success, marker="o")
    axs2[0].set_title("Success Rate (line)")
    axs2[0].set_ylim(0, 1.0)

    axs2[1].plot(x, reward, marker="o")
    axs2[1].set_title("Mean Reward (Success Only)")

    axs2[2].plot(x, steps, marker="o")
    axs2[2].set_title("Mean Steps (Success Only)")

    for ax in axs2:
        ax.set_xticks(x)
        ax.set_xticklabels(pretty_labels, rotation=25, ha="right")

    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, "compare_checkpoints_lines.png")
    fig2.savefig(out_path2, dpi=150)
    print(f"[Compare] Saved line plot: {out_path2}")


def _write_csv(results: Dict[str, Dict[str, float]], out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "compare_checkpoints.csv")
    fields = [
        "checkpoint",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_reward_success",
        "std_reward_success",
        "mean_steps",
        "mean_steps_success",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ckpt, metrics in results.items():
            row = {"checkpoint": ckpt}
            row.update(metrics)
            w.writerow(row)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare offline TD3-BC checkpoints (egocentric eval).")
    parser.add_argument("--config", type=str, default=None, help="Path to evaluate_agent.yaml")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Checkpoint prefixes (without _actor.pth/_critic.pth)",
    )
    parser.add_argument("--out_dir", type=str, default="./agent/logs", help="Output directory for plots/CSV")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode results")
    parser.add_argument("--plot_csv", type=str, default=None, help="Plot only from an existing CSV (skip eval)")
    # Optional: temporary env overrides (no config.py edits)
    parser.add_argument("--static_obstacles", type=int, default=None)
    parser.add_argument("--dynamic_obstacles", type=int, default=None)
    parser.add_argument("--randomize_spawn", action="store_true")
    parser.add_argument("--spawn_min_separation", type=float, default=None)
    parser.add_argument("--randomize_physics", action="store_true")
    args = parser.parse_args()

    if args.plot_csv:
        results = _read_csv(args.plot_csv)
        if not results:
            raise SystemExit(f"[Compare] No rows found in CSV: {args.plot_csv}")
        _try_plot(results, args.out_dir)
        return

    if not args.config or not args.checkpoints:
        raise SystemExit("[Compare] --config and --checkpoints are required unless --plot_csv is provided.")

    eval_cfg = _load_yaml(args.config)
    dataset_path = eval_cfg["dataset_path"]
    hprs_config = eval_cfg["hprs_config"]
    num_episodes = int(eval_cfg.get("num_episodes", 10))
    max_steps = int(eval_cfg.get("max_steps", 2000))
    randomize_seed = bool(eval_cfg.get("randomize_seed", False))
    base_seed = int(eval_cfg.get("base_seed", 0))
    seed_stride = int(eval_cfg.get("seed_stride", 1))

    main_config = Config()
    agent_cfg = main_config.agent_dict
    root = main_config.root_dict
    sim = main_config.simulation_dict
    gym = main_config.gym_environment_dict
    manip_env = main_config.manipulator_environment_dict
    warehouse_env = main_config.warehouse_environment_dict
    observation = main_config.observation_dict
    reward_dict = gym["manipulator_gym_environment"]["reward"]
    gym["num_envs"] = 1
    gym_wh = gym.get("warehouse_gym_environment", {}) or {}
    if args.randomize_spawn:
        gym_wh["randomize_spawn_poses"] = True
    if args.spawn_min_separation is not None:
        gym_wh["spawn_min_separation"] = float(args.spawn_min_separation)
    gym["warehouse_gym_environment"] = gym_wh

    if args.static_obstacles is not None:
        warehouse_env["static_obstacle_count"] = int(args.static_obstacles)
    if args.dynamic_obstacles is not None:
        dyn_cfg = warehouse_env.get("dynamic_obstacles", {}) or {}
        dyn_cfg["dynamic_obstacle_count"] = int(args.dynamic_obstacles)
        warehouse_env["dynamic_obstacles"] = dyn_cfg
    if args.randomize_physics:
        sim["randomize_environment_physics"] = True

    print(
        "[Compare] Env overrides:",
        f"randomize_spawn={gym_wh.get('randomize_spawn_poses')},",
        f"spawn_min_separation={gym_wh.get('spawn_min_separation')},",
        f"static_obstacles={warehouse_env.get('static_obstacle_count')},",
        f"dynamic_obstacles={warehouse_env.get('dynamic_obstacles', {}).get('dynamic_obstacle_count')},",
        f"randomize_physics={sim.get('randomize_environment_physics')}",
    )

    print(f"[Compare] Loading dataset stats: {dataset_path}")
    dataset = ExpertDataset(path=dataset_path)
    stats = dataset.stats

    print(f"[Compare] Creating env with HPRS: {hprs_config}")
    env = get_env(
        agent_cfg, root, sim, gym, manip_env, warehouse_env,
        observation, reward_dict, agent_cfg["log_dir"],
        hprs_config,
        additional_wrapper=EgocentricNormalizationWrapper,
        wrapper_kwargs={
            "stats": stats,
            "normalize_obs": True,
            "unnormalize_actions": True,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[str, Dict[str, float]] = {}
    for ckpt in args.checkpoints:
        label = os.path.basename(ckpt)
        print(f"\n[Compare] Evaluating: {ckpt}")
        agent = TD3BCAgent(
            state_dim=stats["state_dim"],
            action_dim=stats["action_dim"],
            max_action=1.0,
            device=device,
        )
        agent.load(ckpt)
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
        results[label] = metrics
        print(
            f"[{label}] success={metrics['success_rate']:.1%}, "
            f"collision={metrics['collision_rate']:.1%}, "
            f"success_reward={metrics['mean_reward_success']:.2f}, "
            f"success_steps={metrics['mean_steps_success']:.1f}"
        )

    csv_path = _write_csv(results, args.out_dir)
    print(f"[Compare] Saved CSV: {csv_path}")
    _try_plot(results, args.out_dir)

    env.close()


if __name__ == "__main__":
    main()
