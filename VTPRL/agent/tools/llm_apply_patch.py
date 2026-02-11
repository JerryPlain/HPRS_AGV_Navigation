import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _update_constants(cfg: Dict[str, Any], updates: Dict[str, Any]) -> None:
    constants = cfg.get("constants", []) or []
    by_name = {c.get("name"): c for c in constants if isinstance(c, dict)}
    for name, value in updates.items():
        if name in by_name:
            by_name[name]["value"] = float(value)
        else:
            constants.append({"name": name, "value": float(value)})
    cfg["constants"] = constants


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply LLM patch to HPRS YAML")
    parser.add_argument("--base_yaml", type=str, required=True, help="Base HPRS YAML (warehouse.yaml)")
    parser.add_argument("--patch_json", type=str, required=True, help="Patch JSON from LLM")
    parser.add_argument("--out_yaml", type=str, required=True, help="Output YAML path")
    args = parser.parse_args()

    cfg = _load_yaml(args.base_yaml)
    patch = _load_json(args.patch_json)
    constants = patch.get("constants", {})
    _update_constants(cfg, constants)

    out_path = Path(args.out_yaml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(str(out_path), cfg)

    print(f"[LLM] Wrote patched HPRS YAML: {out_path}")
    print(f"[LLM] Applied constants: {constants}")


if __name__ == "__main__":
    main()
