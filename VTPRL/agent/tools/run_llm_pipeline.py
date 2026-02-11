import argparse
import json
import os
import sys
from pathlib import Path

# Add tools dir to path for local imports
tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, tools_dir)

from llm_summarize_run import _read_monitor, _summarize
from llm_propose_patch import _extract_best_patch, ALLOWED_CONSTANTS
from llm_apply_patch import _load_yaml, _write_yaml, _update_constants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM patch pipeline (summary -> patch -> new HPRS).")
    parser.add_argument("--monitor_csv", type=str, required=True)
    parser.add_argument("--base_yaml", type=str, required=True)
    parser.add_argument("--out_yaml", type=str, required=True)
    parser.add_argument("--summary_json", type=str, default="./agent/logs/run_summary.json")
    parser.add_argument("--patch_json", type=str, default="./agent/logs/llm_patch.json")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--avoid_params", type=str, default="", help="Comma-separated params to avoid")
    parser.add_argument("--feedback", type=str, default="", help="Reason from previous rejection.")
    args = parser.parse_args()

    # 1) summarize
    rows = _read_monitor(args.monitor_csv)
    summary = _summarize(rows, args.window)
    summary["monitor_csv"] = args.monitor_csv
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[LLM] summary -> {args.summary_json}")
    print(
        "[LLM] Summary:",
        f"episodes={summary['episodes']}",
        f"success_rate={summary['success_rate']:.3f}",
        f"collision_rate={summary.get('collision_rate', float('nan')):.3f}",
        f"mean_success_reward={summary.get('mean_success_reward', float('nan')):.2f}",
        f"mean_reward={summary['mean_reward']:.2f}",
        f"mean_length={summary['mean_length']:.1f}",
    )

    # 2) propose patch via HF model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    base_cfg = _load_yaml(args.base_yaml)
    base_constants = {}
    for item in base_cfg.get("constants", []) or []:
        name = item.get("name")
        if name is None:
            continue
        base_constants[name] = item.get("value")
    avoid_params = [p.strip() for p in args.avoid_params.split(",") if p.strip()]
    avoid_txt = ", ".join(avoid_params) if avoid_params else "none"

    feedback_txt = args.feedback.strip()
    feedback_block = f"Previous rejection reason: {feedback_txt}\n\n" if feedback_txt else ""

    prompt = (
        "You are tuning reward shaping constants for a navigation RL task.\n\n"
        "Primary goal: increase success_rate if possible.\n"
        "Secondary goal: do NOT increase collision_rate.\n"
        "Mean reward is not a hard constraint.\n\n"
        "Rules:\n"
        "1) You MUST modify exactly 2 constants: one progress-related and one safety-related.\n"
        "   - Progress-related: approach_dist, delta_dist_weight\n"
        "   - Safety-related: laser_safe, collision_penalty\n"
        "2) Each change must be within ±40% of current value.\n"
        f"3) Do NOT modify any of these recently changed params: {avoid_txt}.\n\n"
        "Output ONLY JSON wrapped between BEGIN_JSON and END_JSON.\n"
        "Do NOT include any extra text, markdown, or code fences.\n"
        "Keep explanation to 2–4 short sentences.\n"
        "The explanation MUST explicitly analyze the Summary below.\n"
        "If a previous rejection reason is provided, explain why it failed and how this change avoids that failure.\n"
        "BEGIN_JSON\n"
        "{\n"
        '  "constants": { "param_name": 1.0, "param_name_2": 2.0 },\n'
        '  "explanation": "2–4 sentences. Must cite Summary numbers and, if provided, address the previous rejection reason."\n'
        "}\n"
        "END_JSON\n"
        "Constraints:\n"
        f"- Only use these constants: {ALLOWED_CONSTANTS}\n"
        "- Changes should be within ±40%.\n\n"
        f"Current constants:\n{json.dumps(base_constants, indent=2)}\n\n"
        f"Summary:\n{json.dumps(summary, indent=2)}\n\n"
        f"{feedback_block}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("[LLM] generated_text:\n" + text)
    patch = _extract_best_patch(text, ALLOWED_CONSTANTS)
    constants = patch.get("constants", {})
    explanation = str(patch.get("explanation", "")).strip()
    needs_evidence = True
    filtered = {k: v for k, v in constants.items() if k in ALLOWED_CONSTANTS}
    # Drop tiny/no-op changes (<5%) to avoid meaningless patches
    filtered_nontrivial = {}
    for k, v in filtered.items():
        base_v = base_constants.get(k)
        if base_v is None:
            continue
        if base_v == 0:
            if v != 0:
                filtered_nontrivial[k] = v
            continue
        if abs((v - base_v) / base_v) >= 0.05:
            filtered_nontrivial[k] = v
    patch["constants"] = filtered_nontrivial

    # Require evidence-backed explanation: must include at least one numeric reference
    if needs_evidence:
        has_number = any(ch.isdigit() for ch in explanation)
        if not has_number:
            patch["constants"] = {}
            patch["explanation"] = "rejected: explanation missing numeric evidence (include at least one number from the summary)."

    Path(args.patch_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.patch_json, "w", encoding="utf-8") as f:
        json.dump(patch, f, indent=2)
    print(f"[LLM] patch -> {args.patch_json}")
    print(f"[LLM] Patch constants: {filtered}")

    # 3) apply patch to YAML
    cfg = _load_yaml(args.base_yaml)
    _update_constants(cfg, filtered)
    Path(args.out_yaml).parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(args.out_yaml, cfg)
    print(f"[LLM] new HPRS -> {args.out_yaml}")


if __name__ == "__main__":
    main()
