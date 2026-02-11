# LLM-Online HPRS Experiment Report
## 1. Goal
Use an LLM to adapt HPRS reward shaping constants during online TD3 training to improve success rate, while keeping collision rate from getting worse. The LLM does not train the policy; it only proposes small, controlled HPRS adjustments.

## 2. Design Rationale
- Training is already heavy. Putting an LLM inside the training loop is slow and unstable.
- The LLM is better used as an external tuner that reads concise summaries and proposes small parameter changes.
- This keeps the process controllable, explainable, and reproducible.

## 3. End-to-End Workflow
Each run is divided into multiple segments. For each segment:

1) Train for a short segment.
2) Summarize results (monitor.csv -> run_summary.json).
3) LLM proposes a patch (limited to 2 constants).
4) Apply patch to generate a new HPRS YAML.
5) Validate old vs new HPRS.
6) Accept or reject based on a clear rule.

All outputs are logged per segment.

## 4. Inputs and Outputs

### Training (per segment)
Script: `agent/tools/run_online_llm_loop.py`
- Input: `online_td3_llm.yaml` (segment_steps, pretrained path, etc.)
- Output: checkpoints, `monitor.csv`

### Summary
Script: `agent/tools/llm_summarize_run.py`
- Input: `monitor.csv`
- Output: `run_summary.json` (success_rate, collision_rate, mean_reward, mean_length, etc.)

### Patch Proposal
Script: `agent/tools/run_llm_pipeline.py`
- Input: current HPRS, `run_summary.json`, optional previous reject reason
- Output: `llm_patch.json`

### Apply Patch
Script: `agent/tools/llm_apply_patch.py`
- Input: patch + base HPRS
- Output: `warehouse_llm_seg_xx.yaml`

### Validation
Script: `agent/tools/run_online_llm_loop.py`
- Input: old HPRS vs new HPRS, same model
- Output: acceptance decision + reject logs (if any)

## 5. Acceptance Rule
Current rule:
- success_rate must not drop
- collision_rate must not increase
- if success_rate and collision_rate are unchanged, mean_reward must improve

This is implemented in `agent/tools/run_online_llm_loop.py`.

## 6. LLM Patch Constraints
Prompt constraints:
- Modify exactly 2 constants (1 progress + 1 safety)
- Each change within +/-40%
- Prefer small changes (10-20%) unless success_rate is very low
- Must output JSON only, wrapped by BEGIN_JSON / END_JSON
- Must include a short explanation grounded in the summary

Implementation: `agent/tools/llm_propose_patch.py` and `agent/tools/run_llm_pipeline.py`.

## 7. Failure Feedback Loop
If a patch is rejected:
- Reject reason is computed and written to `reject_reason.txt`
- The reason is fed into the next LLM prompt
This avoids repeating known-bad directions.

## 8. Logging and Artifacts
Per segment:
- `online_td3_seg.yaml`
- `monitor.csv`
- `run_summary.json`
- `llm_patch.json`
- `warehouse_llm_seg_xx.yaml`
- `hprs_diff.json`
- `reject.json` / `reject_reason.txt` (if rejected)

Global:
- `agent/logs/llm_runs/llm_loop_summary.json`

## 9. How to Run

### Baseline Online
```
python agent/td3/online_td3/train_online_td3.py \
  --mode warm_start \
  --config agent/config/online_td3_baseline.yaml
```

### LLM-Online
```
python agent/tools/run_online_llm_loop.py \
  --base_config agent/config/online_td3_llm.yaml \
  --segments 10 \
  --segment_steps 10000 \
  --llm_model Qwen/Qwen2.5-3B-Instruct \
  --val_episodes 10 \
  --accept_delta 0 \
  --accept_reward_delta 0.1 \
  --accept_collision_delta 0 \
  --val_verbose
```

## 10. Evaluation (Separate HPRS per Model)

Config: `agent/config/evaluate_compare_models.yaml`  
Script: `agent/tools/compare_models_with_hprs.py`

```
python agent/tools/compare_models_with_hprs.py \
  --config agent/config/evaluate_compare_models.yaml \
  --verbose
```

Outputs:
- `agent/logs/compare_models/compare_models.csv`
- `agent/logs/compare_models/compare_models_bars.png`

## 11. Concrete Results

### 11.1 Final Comparison (Baseline vs LLM)

Source: `agent/logs/compare_models/compare_models.csv`
Chart: `agent/logs/compare_models/compare_models_bars.png`

### 11.2 Final HPRS Changes (Baseline → LLM Final)

Comparing `auto-shaping/configs/warehouse.yaml` vs  
`agent/logs/llm_runs/seg_09/warehouse_llm_seg_09.yaml`:

- `approach_dist`: **2.0 → 0.96**
- `collision_penalty`: **1.0 → 1.4**
- `delta_dist_weight`: **1.0 → 1.2**

## 12. Improvement Process (Segment-by-Segment)

This summarizes each segment’s HPRS changes and accept/reject outcome:

- **seg_01** (accepted)  
  `approach_dist: 2.0 → 1.6`, `delta_dist_weight: 1.0 → 1.2`
- **seg_02** (rejected)  
  no changes (rejected: success_rate dropped)
- **seg_03** (rejected)  
  no changes (rejected: success_rate dropped)
- **seg_04** (accepted)  
  no changes (kept previous)
- **seg_05** (rejected)  
  `collision_penalty: 1.0 → 0.9` (rejected: success_rate dropped)
- **seg_06** (rejected)  
  `approach_dist: 1.6 → 2.24`, `delta_dist_weight: 1.2 → 1.68`, `laser_safe: 1.5 → 0.9`
- **seg_07** (rejected)  
  `delta_dist_weight: 1.2 → 1.72` (rejected: mean_reward not improved)
- **seg_08** (accepted)  
  no changes (kept previous)
- **seg_09** (accepted)  
  `approach_dist: 1.6 → 0.96`, `collision_penalty: 1.0 → 1.4`
- **seg_10** (rejected)  
  `delta_dist_weight: 1.2 → 1.24` (rejected: success_rate dropped)

## 13. Notes on Interpretation

- The LLM policy improved success rate (0.48 → 0.76) but at a cost of higher collision rate (0.01 → 0.035).
- Mean reward decreased for the LLM model, while mean success reward improved slightly.
- If success rate is the primary target, the LLM‑tuned HPRS is favorable; if safety or reward is critical, tighten acceptance constraints.

## 14. Summary
This system enables safe and interpretable HPRS tuning during online RL by:
- keeping changes small and traceable
- validating each proposed update
- using LLMs as a controlled outer‑loop optimizer

The result is a structured, reproducible pipeline for LLM‑driven reward shaping.