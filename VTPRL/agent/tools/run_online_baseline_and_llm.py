import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n[Runner] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run online TD3 baseline, then run LLM-assisted online TD3 loop."
    )
    parser.add_argument(
        "--baseline_config",
        type=str,
        default="agent/config/online_td3_baseline.yaml",
        help="Path to online baseline config.",
    )
    parser.add_argument(
        "--llm_config",
        type=str,
        default="agent/config/online_td3_llm.yaml",
        help="Path to online LLM base config.",
    )
    parser.add_argument("--mode", type=str, default="warm_start", help="TD3 mode.")
    parser.add_argument("--segments", type=int, default=5, help="Number of LLM segments.")
    parser.add_argument("--segment_steps", type=int, default=20000, help="Steps per segment.")
    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model id for LLM.",
    )
    parser.add_argument("--val_episodes", type=int, default=15, help="Validation episodes.")
    parser.add_argument("--accept_delta", type=float, default=0.0, help="Success tolerance.")
    parser.add_argument(
        "--accept_reward_delta",
        type=float,
        default=0.1,
        help="Reward improvement required when success/collision are unchanged.",
    )
    parser.add_argument(
        "--accept_collision_delta",
        type=float,
        default=0.0,
        help="Collision tolerance.",
    )
    parser.add_argument(
        "--val_verbose",
        action="store_true",
        help="Print per-episode validation results.",
    )
    args = parser.parse_args()

    baseline_cfg = Path(args.baseline_config)
    llm_cfg = Path(args.llm_config)
    if not baseline_cfg.exists():
        raise FileNotFoundError(f"baseline_config not found: {baseline_cfg}")
    if not llm_cfg.exists():
        raise FileNotFoundError(f"llm_config not found: {llm_cfg}")

    _run(
        [
            sys.executable,
            "agent/td3/online_td3/train_online_td3.py",
            "--mode",
            args.mode,
            "--config",
            str(baseline_cfg),
        ]
    )

    llm_cmd = [
        sys.executable,
        "agent/tools/run_online_llm_loop.py",
        "--base_config",
        str(llm_cfg),
        "--segments",
        str(args.segments),
        "--segment_steps",
        str(args.segment_steps),
        "--llm_model",
        args.llm_model,
        "--val_episodes",
        str(args.val_episodes),
        "--accept_delta",
        str(args.accept_delta),
        "--accept_reward_delta",
        str(args.accept_reward_delta),
        "--accept_collision_delta",
        str(args.accept_collision_delta),
    ]
    if args.val_verbose:
        llm_cmd.append("--val_verbose")
    _run(llm_cmd)


if __name__ == "__main__":
    main()
