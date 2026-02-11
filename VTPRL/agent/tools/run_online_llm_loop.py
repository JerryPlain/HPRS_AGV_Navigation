import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml

# Add agent/ to path so "from config import Config" works when run from repo root
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def _load_constants(path: str) -> Dict[str, float]:
    data = _load_yaml(path) or {}
    consts = {}
    for item in data.get("constants", []) or []:
        name = item.get("name")
        if name is None:
            continue
        consts[name] = item.get("value")
    return consts

def _diff_constants(old_yaml: str, new_yaml: str) -> Dict[str, Dict[str, Any]]:
    old_c = _load_constants(old_yaml)
    new_c = _load_constants(new_yaml)
    keys = sorted(set(old_c) | set(new_c))
    diff: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        if old_c.get(k) != new_c.get(k):
            diff[k] = {"old": old_c.get(k), "new": new_c.get(k)}
    return diff


def _evaluate_metrics(
    hprs_config: str,
    model_prefix: str,
    dataset_path: str,
    agent_cfg: Dict[str, Any],
    root: Dict[str, Any],
    sim: Dict[str, Any],
    gym: Dict[str, Any],
    manip_env: Dict[str, Any],
    warehouse_env: Dict[str, Any],
    observation: Dict[str, Any],
    reward_dict: Dict[str, Any],
    log_dir: str,
    episodes: int,
    max_steps: int,
    base_seed: int,
    seed_stride: int,
    verbose: bool,
) -> Dict[str, float]:
    # Local imports to keep startup light
    from config import Config  # noqa: F401
    from envs.get_env import get_env
    from expert.expert_dataset import ExpertDataset
    from td3.td3bc.td3_bc_agent import TD3BCAgent
    from td3.online_td3.egocentric_normalization_wrapper import EgocentricNormalizationWrapper

    dataset = ExpertDataset(path=dataset_path)
    stats = dataset.stats

    # Ensure single env for eval
    gym = dict(gym)
    gym["num_envs"] = 1

    agent_cfg = dict(agent_cfg)
    agent_cfg["log_dir"] = log_dir

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

    successes: list[float] = []
    collisions: list[float] = []
    rewards: list[float] = []
    success_rewards: list[float] = []
    for ep in range(episodes):
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
        successes.append(1.0 if success else 0.0)
        collisions.append(1.0 if collision else 0.0)
        rewards.append(total_reward)
        if success:
            success_rewards.append(total_reward)
        if verbose:
            print(
                f"[Val] ep={ep+1}/{episodes} "
                f"success={int(success)} collision={int(collision)} reward={total_reward:.2f}"
            )

    env.close()
    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_success_reward": float(np.mean(success_rewards)) if success_rewards else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online TD3 with LLM-driven HPRS updates.")
    parser.add_argument("--base_config", type=str, required=True, help="Base online TD3 YAML")
    parser.add_argument("--segments", type=int, default=5, help="Number of LLM update segments")
    parser.add_argument("--segment_steps", type=int, default=20000, help="Steps per segment")
    parser.add_argument("--monitor_csv", type=str, default=None, help="Override monitor.csv path")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--summary_window", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="./agent/logs/llm_runs")
    parser.add_argument("--val_episodes", type=int, default=10, help="Validation episodes per patch")
    parser.add_argument("--val_max_steps", type=int, default=2000, help="Max steps per validation episode")
    parser.add_argument("--val_base_seed", type=int, default=1000)
    parser.add_argument("--val_seed_stride", type=int, default=1)
    parser.add_argument(
        "--accept_delta",
        type=float,
        default=0.0,
        help="Tolerance for success_rate comparisons (0.0 = must not drop).",
    )
    parser.add_argument(
        "--accept_reward_delta",
        type=float,
        default=0.2,
        help="Minimum mean-reward gain required only when success/collision are unchanged.",
    )
    parser.add_argument(
        "--accept_collision_delta",
        type=float,
        default=0.0,
        help="Tolerance for collision_rate comparisons (0.0 = must not increase).",
    )
    parser.add_argument("--val_verbose", action="store_true", help="Print per-episode validation results")
    args = parser.parse_args()

    base = _load_yaml(args.base_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start with base HPRS config
    current_hprs = base.get("hprs_config", "auto-shaping/configs/warehouse.yaml")
    pretrained = base.get("pretrained_path", None)
    models_dir = base.get("models_dir", "./agent/logs/models/online_llm")

    # Load env config once for validation
    from config import Config
    cfg = Config()
    agent_cfg = cfg.agent_dict
    root = cfg.root_dict
    sim = cfg.simulation_dict
    gym = cfg.gym_environment_dict
    manip_env = cfg.manipulator_environment_dict
    warehouse_env = cfg.warehouse_environment_dict
    observation = cfg.observation_dict
    reward_dict = gym["manipulator_gym_environment"]["reward"]

    dataset_path = base.get("dataset_path", "./agent/logs/expert_dataset.pkl")
    last_feedback = ""

    for seg in range(args.segments):
        seg_id = seg + 1
        seg_dir = out_dir / f"seg_{seg_id:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Write per-segment config
        seg_cfg = dict(base)
        seg_cfg["max_timesteps"] = int(args.segment_steps)
        seg_cfg["hprs_config"] = current_hprs
        if pretrained:
            seg_cfg["pretrained_path"] = pretrained
        seg_cfg["models_dir"] = str(Path(models_dir) / f"seg_{seg_id:02d}")
        seg_cfg["log_dir"] = str(seg_dir / "runs")

        cfg_path = seg_dir / "online_td3_seg.yaml"
        _write_yaml(str(cfg_path), seg_cfg)

        print(f"\n[LLM-Loop] Segment {seg_id}/{args.segments}")
        print(f"[LLM-Loop]   hprs_config: {current_hprs}")
        print(f"[LLM-Loop]   config: {cfg_path}")
        print(f"[LLM-Loop]   train log_dir: {seg_cfg['log_dir']}")

        # Train segment
        subprocess.run(
            [
                "python",
                "agent/td3/online_td3/train_online_td3.py",
                "--mode",
                "warm_start",
                "--config",
                str(cfg_path),
            ],
            check=True,
        )

        # Determine monitor.csv (VecMonitor writes to agent_cfg["log_dir"])
        default_monitor = Path("agent/logs/monitor.csv")
        seg_monitor = seg_dir / "runs" / "monitor.csv"
        monitor_candidates = []
        if args.monitor_csv:
            monitor_candidates.append(Path(args.monitor_csv))
        monitor_candidates.extend([seg_monitor, default_monitor])

        monitor_csv = None
        for cand in monitor_candidates:
            if cand.exists():
                monitor_csv = str(cand)
                break
        if monitor_csv is None:
            # fallback: pick newest monitor.csv under agent/logs
            log_root = Path("agent/logs")
            found = sorted(log_root.rglob("monitor.csv"), key=lambda p: p.stat().st_mtime, reverse=True) if log_root.exists() else []
            if found:
                monitor_csv = str(found[0])
            else:
                raise FileNotFoundError("monitor.csv not found. Check log_dir or VecMonitor output.")
        print(f"[LLM-Loop]   train monitor_csv: {monitor_csv}")

        # Summarize + propose patch + apply patch
        summary_json = seg_dir / "run_summary.json"
        patch_json = seg_dir / "llm_patch.json"
        next_hprs = seg_dir / f"warehouse_llm_seg_{seg_id:02d}.yaml"

        # Avoid reusing last changed params (if any)
        avoid_params: list[str] = []
        if seg_id > 1:
            prev_diff = out_dir / f"seg_{seg_id-1:02d}" / "hprs_diff.json"
            if prev_diff.exists():
                try:
                    with open(prev_diff, "r", encoding="utf-8") as f:
                        prev = json.load(f)
                    avoid_params = list(prev.keys())
                except Exception:
                    avoid_params = []

        subprocess.run(
            [
                "python",
                "agent/tools/run_llm_pipeline.py",
                "--monitor_csv",
                monitor_csv,
                "--base_yaml",
                current_hprs,
                "--out_yaml",
                str(next_hprs),
                "--summary_json",
                str(summary_json),
                "--patch_json",
                str(patch_json),
                "--model_id",
                args.llm_model,
                "--avoid_params",
                ",".join(avoid_params),
                "--feedback",
                last_feedback,
                "--window",
                str(args.summary_window),
            ],
            check=True,
        )
        # Record diff vs previous HPRS
        diff = _diff_constants(current_hprs, str(next_hprs))
        diff_path = seg_dir / "hprs_diff.json"
        with open(diff_path, "w", encoding="utf-8") as f:
            json.dump(diff, f, indent=2)
        if diff:
            print(f"[LLM-Loop]   hprs diff: {diff_path}")
            for k, v in diff.items():
                print(f"[LLM-Loop]     {k}: {v['old']} -> {v['new']}")
        else:
            print(f"[LLM-Loop]   hprs diff: no changes")
            last_feedback = "previous patch had no effective changes (values too close to current)."

        # Update for next segment
        last_prefix = str(Path(seg_cfg["models_dir"]) / "TD3_warm_start_Online_EGO_last")

        # Validate old vs new HPRS before accepting
        print("[LLM-Loop] Validation start")
        print(f"[LLM-Loop]   val episodes: {args.val_episodes}")
        print(f"[LLM-Loop]   val max_steps: {args.val_max_steps}")
        print(f"[LLM-Loop]   old HPRS: {current_hprs}")
        print(f"[LLM-Loop]   new HPRS: {next_hprs}")
        val_log_dir = str(seg_dir / "val_runs")
        old_metrics = _evaluate_metrics(
            hprs_config=current_hprs,
            model_prefix=last_prefix,
            dataset_path=dataset_path,
            agent_cfg=agent_cfg,
            root=root,
            sim=sim,
            gym=gym,
            manip_env=manip_env,
            warehouse_env=warehouse_env,
            observation=observation,
            reward_dict=reward_dict,
            log_dir=val_log_dir,
            episodes=args.val_episodes,
            max_steps=args.val_max_steps,
            base_seed=args.val_base_seed,
            seed_stride=args.val_seed_stride,
            verbose=args.val_verbose,
        )
        new_metrics = _evaluate_metrics(
            hprs_config=str(next_hprs),
            model_prefix=last_prefix,
            dataset_path=dataset_path,
            agent_cfg=agent_cfg,
            root=root,
            sim=sim,
            gym=gym,
            manip_env=manip_env,
            warehouse_env=warehouse_env,
            observation=observation,
            reward_dict=reward_dict,
            log_dir=val_log_dir,
            episodes=args.val_episodes,
            max_steps=args.val_max_steps,
            base_seed=args.val_base_seed,
            seed_stride=args.val_seed_stride,
            verbose=args.val_verbose,
        )
        print(f"[LLM-Loop]   val log_dir: {val_log_dir}")

        print(
            "[LLM-Loop] Validation metrics:",
            f"old_sr={old_metrics['success_rate']:.3f}",
            f"old_col={old_metrics['collision_rate']:.3f}",
            f"old_rew={old_metrics['mean_reward']:.2f}",
            f"| new_sr={new_metrics['success_rate']:.3f}",
            f"new_col={new_metrics['collision_rate']:.3f}",
            f"new_rew={new_metrics['mean_reward']:.2f}",
        )

        sr_old = old_metrics["success_rate"]
        sr_new = new_metrics["success_rate"]
        col_old = old_metrics["collision_rate"]
        col_new = new_metrics["collision_rate"]
        rew_old = old_metrics["mean_reward"]
        rew_new = new_metrics["mean_reward"]

        success_not_worse = sr_new >= sr_old - args.accept_delta
        collision_not_worse = col_new <= col_old + args.accept_collision_delta
        sr_same = abs(sr_new - sr_old) <= args.accept_delta
        col_same = abs(col_new - col_old) <= args.accept_collision_delta
        reward_improved = rew_new >= rew_old + args.accept_reward_delta

        needs_reward = sr_same and col_same
        accept = success_not_worse and collision_not_worse and (not needs_reward or reward_improved)

        if accept:
            current_hprs = str(next_hprs)
            print("[LLM-Loop] Patch accepted.")
            last_feedback = ""
        else:
            print("[LLM-Loop] Patch rejected. Keeping previous HPRS.")
            sr_delta = sr_new - sr_old
            col_delta = col_new - col_old
            rew_delta = rew_new - rew_old
            reasons = []
            if sr_new < sr_old - args.accept_delta:
                reasons.append(f"success_rate dropped by {sr_delta:.3f}")
            if col_new > col_old + args.accept_collision_delta:
                reasons.append(f"collision_rate increased by {col_delta:.3f}")
            if needs_reward and not reward_improved:
                reasons.append(f"mean_reward did not improve ({rew_delta:.2f})")
            if not reasons:
                reasons.append("did not meet acceptance criteria")
            last_feedback = " ; ".join(reasons)
            reject_path = seg_dir / "reject.json"
            with open(reject_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "old_hprs": current_hprs,
                        "new_hprs": str(next_hprs),
                        "old_metrics": old_metrics,
                        "new_metrics": new_metrics,
                        "reject_reason": last_feedback,
                    },
                    f,
                    indent=2,
                )
            print(f"[LLM-Loop] Wrote reject info: {reject_path}")
            reject_reason_path = seg_dir / "reject_reason.txt"
            with open(reject_reason_path, "w", encoding="utf-8") as f:
                f.write(last_feedback + "\n")
            print(f"[LLM-Loop] Reject reason: {last_feedback}")
        print(
            "[LLM-Loop] Decision:",
            f"success_not_worse={success_not_worse}",
            f"collision_not_worse={collision_not_worse}",
            f"needs_reward={needs_reward}",
            f"reward_improved={reward_improved}",
            f"(accept_delta={args.accept_delta}, "
            f"accept_reward_delta={args.accept_reward_delta}, "
            f"accept_collision_delta={args.accept_collision_delta})",
        )

        # Save a copy of the accepted hprs for traceability
        accepted_copy = seg_dir / "accepted_hprs.yaml"
        try:
            shutil.copyfile(current_hprs, accepted_copy)
        except Exception:
            pass

        pretrained = last_prefix

    summary = {
        "segments": args.segments,
        "last_hprs": current_hprs,
        "last_checkpoint": pretrained,
        "val_episodes": args.val_episodes,
        "accept_delta": args.accept_delta,
        "accept_reward_delta": args.accept_reward_delta,
        "accept_collision_delta": args.accept_collision_delta,
        "runs_dir": str(out_dir),
    }
    summary_path = out_dir / "llm_loop_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n[LLM-Loop] Done.")
    print(f"[LLM-Loop] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
