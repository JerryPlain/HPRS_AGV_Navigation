import argparse
import json
from pathlib import Path
from typing import Dict, Any
import re


ALLOWED_CONSTANTS = [
    "approach_dist",
    "delta_dist_weight",
    "laser_safe",
    "yaw_comfort",
    "pos_comfort",
    "near_goal",
    "very_near_goal",
    "v_comfort",
    "omega_comfort",
    "collision_penalty",
]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_json(text: str) -> Dict[str, Any]:
    stack = []
    candidates: list[Dict[str, Any]] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            snippet = text[start : i + 1]
            try:
                parsed = json.loads(snippet)
            except Exception:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
    if not candidates:
        raise ValueError("No valid JSON object found in model output.")
    # Prefer the last JSON object that contains constants
    for parsed in reversed(candidates):
        if "constants" in parsed:
            return parsed
    return candidates[-1]

def _fallback_from_text(text: str, allowed: list[str]) -> Dict[str, Any]:
    # Try to extract "Approach Dist: Increase from 1.5 to 2.1" style lines
    mappings = {
        "approach_dist": r"Approach Dist.*?to\s+([0-9.+-]+)",
        "delta_dist_weight": r"Delta Dist Weight.*?to\s+([0-9.+-]+)",
        "laser_safe": r"Laser Safe.*?to\s+([0-9.+-]+)",
        "collision_penalty": r"Collision Penalty.*?to\s+([0-9.+-]+)",
    }
    constants: Dict[str, Any] = {}
    for key, pat in mappings.items():
        if key not in allowed:
            continue
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                constants[key] = float(m.group(1))
            except Exception:
                pass
    if constants:
        return {"constants": constants, "explanation": "extracted from text fallback"}
    return {}


def _extract_best_patch(text: str, allowed: list[str]) -> Dict[str, Any]:
    # Prefer explicit BEGIN_JSON/END_JSON block if present
    if "BEGIN_JSON" in text and "END_JSON" in text:
        start = text.find("BEGIN_JSON") + len("BEGIN_JSON")
        end = text.find("END_JSON")
        snippet = text[start:end].strip()
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    stack = []
    candidates: list[Dict[str, Any]] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            snippet = text[start : i + 1]
            try:
                parsed = json.loads(snippet)
            except Exception:
                continue
            if isinstance(parsed, dict) and "constants" in parsed:
                candidates.append(parsed)
    if not candidates:
        fallback = _fallback_from_text(text, allowed)
        if fallback:
            return fallback
        return _extract_json(text)
    # Prefer a patch that contains at least one allowed constant, choose the last one
    for parsed in reversed(candidates):
        consts = parsed.get("constants", {})
        if any(k in allowed for k in consts.keys()):
            return parsed
    return candidates[-1]


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _extract_constants(cfg: Dict[str, Any]) -> Dict[str, Any]:
    consts = {}
    for item in cfg.get("constants", []) or []:
        name = item.get("name")
        if name is None:
            continue
        consts[name] = item.get("value")
    return consts

def _filter_changes(
    proposed: Dict[str, Any],
    base_constants: Dict[str, Any],
    min_ratio: float = 0.05,
) -> Dict[str, Any]:
    """Drop proposed changes that are too small to matter."""
    filtered: Dict[str, Any] = {}
    for k, v in proposed.items():
        if k not in base_constants:
            continue
        old = base_constants[k]
        try:
            old_f = float(old)
            new_f = float(v)
        except Exception:
            continue
        denom = abs(old_f) if abs(old_f) > 1e-8 else 1.0
        if abs(new_f - old_f) / denom < min_ratio:
            continue
        filtered[k] = v
    return filtered

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM patch proposal for HPRS constants")
    parser.add_argument("--summary_json", type=str, required=True, help="Input run_summary.json")
    parser.add_argument("--out_patch", type=str, required=True, help="Output patch JSON")
    parser.add_argument("--base_yaml", type=str, required=True, help="Base HPRS yaml for current constants")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--avoid_params", type=str, default="", help="Comma-separated params to avoid")
    args = parser.parse_args()

    summary = _load_json(args.summary_json)
    base_cfg = _load_yaml(args.base_yaml)
    base_constants = _extract_constants(base_cfg)
    avoid_params = [p.strip() for p in args.avoid_params.split(",") if p.strip()]
    avoid_txt = ", ".join(avoid_params) if avoid_params else "none"

    prompt = (
        "You are tuning HPRS reward shaping constants for a navigation RL task.\n\n"
        "Task goal: reach the goal position. Being closer to the goal is always better.\n\n"
        "Primary goal: increase success_rate.\n"
        "Secondary goal: do NOT increase collision_rate.\n"
        "Mean reward is not a hard constraint.\n"
        "Use the summary metrics explicitly in your reasoning (success_rate, collision_rate, mean_success_reward).\n\n"
        "Rules:\n"
        "1) You MUST modify exactly 2 constants: one progress-related and one safety-related.\n"
        "   - Progress-related: approach_dist, delta_dist_weight\n"
        "   - Safety-related: laser_safe, collision_penalty\n"
        "2) Each change must be within ±40% of current value.\n"
        "3) Prefer small changes (10–20%) unless success_rate is very low (<0.4).\n"
        f"4) Do NOT modify any of these recently changed params: {avoid_txt}.\n\n"
        "Return ONLY a single JSON object and nothing else.\n"
        "Do NOT include code fences or extra text.\n"
        "Wrap the JSON between BEGIN_JSON and END_JSON exactly.\n"
        "BEGIN_JSON\n"
        "{\n"
        '  "constants": {"param_name": 1.0, "param_name_2": 2.0},\n'
        '  "explanation": "2–4 sentences. Must cite summary numbers (e.g., success_rate=0.50, collision_rate=0.20) and explain why these changes help."\n'
        "}\n"
        "END_JSON\n\n"
        f"Allowed constants: {ALLOWED_CONSTANTS}\n\n"
        f"Current constants:\n{json.dumps(base_constants, indent=2)}\n\n"
        f"Summary (must cite these numbers in explanation):\n{json.dumps(summary, indent=2)}\n"
    )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:
        raise SystemExit(
            "[LLM] transformers not available. Install with: pip install transformers"
        ) from exc

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
    filtered = {k: v for k, v in constants.items() if k in ALLOWED_CONSTANTS}
    filtered = _filter_changes(filtered, base_constants, min_ratio=0.05)
    patch["constants"] = filtered

    out_path = Path(args.out_patch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(patch, f, indent=2)

    print(f"[LLM] Wrote patch: {out_path}")
    print(f"[LLM] Patch constants: {filtered}")


if __name__ == "__main__":
    main()
