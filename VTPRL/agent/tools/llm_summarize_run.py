import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _read_monitor(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            reader = csv.DictReader([line]) if line.startswith("r,") else None
            if reader is not None:
                # header line, skip
                continue
            break

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for row in reader:
            rows.append(row)
    return rows


def _summarize(rows: List[Dict[str, str]], window: int) -> Dict[str, float]:
    if not rows:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "mean_reward": 0.0,
            "mean_length": 0.0,
            "reward_std": 0.0,
            "length_std": 0.0,
            "mean_success_reward": 0.0,
        }

    tail = rows[-window:] if window > 0 else rows
    rewards = np.array([float(r["r"]) for r in tail], dtype=np.float32)
    lengths = np.array([float(r["l"]) for r in tail], dtype=np.float32)
    successes = np.array([1.0 if str(r.get("success", "")).lower() == "true" else 0.0 for r in tail], dtype=np.float32)
    collisions = np.array(
        [1.0 if str(r.get("collision", "")).lower() == "true" else 0.0 for r in tail],
        dtype=np.float32,
    )
    timeouts = np.array(
        [1.0 if str(r.get("timeout", "")).lower() == "true" else 0.0 for r in tail],
        dtype=np.float32,
    )
    success_rewards = rewards[successes > 0.0]
    has_collision = "collision" in tail[0]
    has_timeout = "timeout" in tail[0]

    return {
        "episodes": int(len(tail)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)) if has_collision else 0.0,
        "timeout_rate": float(np.mean(timeouts)) if has_timeout else 0.0,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "reward_std": float(np.std(rewards)),
        "length_std": float(np.std(lengths)),
        "mean_success_reward": float(np.mean(success_rewards)) if success_rewards.size > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize monitor.csv into run_summary.json")
    parser.add_argument("--monitor_csv", type=str, required=True, help="Path to monitor.csv")
    parser.add_argument("--out_json", type=str, required=True, help="Output summary JSON")
    parser.add_argument("--window", type=int, default=100, help="Episodes to summarize (tail window)")
    args = parser.parse_args()

    rows = _read_monitor(args.monitor_csv)
    summary = _summarize(rows, args.window)
    summary["monitor_csv"] = args.monitor_csv

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[LLM] Wrote summary: {out_path}")
    print(
        "[LLM] Summary:",
        f"episodes={summary['episodes']}",
        f"success_rate={summary['success_rate']:.3f}",
        f"mean_reward={summary['mean_reward']:.2f}",
        f"mean_length={summary['mean_length']:.1f}",
    )


if __name__ == "__main__":
    main()
