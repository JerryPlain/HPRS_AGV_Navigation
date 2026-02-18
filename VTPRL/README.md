# LLM-Online HPRS Experiment Report

## 1. Executive Summary
This report documents an online TD3 training pipeline where an LLM performs constrained outer-loop tuning of HPRS reward constants for AGV warehouse navigation.

Key outcome (from `agent/logs/compare_models/compare_models.csv`):
- Success rate: `0.48 -> 0.76` (`+0.28` absolute)
- Collision rate: `0.010 -> 0.035` (`+0.025` absolute)
- Mean reward: `-11.41 -> -78.74`
- Mean reward on successful episodes: `2.07 -> 2.38`
- Mean episode steps: `1362.98 -> 938.81`

Interpretation: the LLM-tuned configuration improves task completion and shortens episodes, but safety (collision rate) degrades under the final evaluation configuration.

## 2. Objective
The objective is to optimize online performance by adapting HPRS constants during training segments, while keeping the update process:
- constrained (small, explicit parameter edits),
- auditable (every patch and decision logged),
- reversible (rejected patches do not propagate).

The LLM does not train the policy network; it proposes reward-constant patches only.

## 3. Method Overview
Each run is split into fixed-length segments. For each segment:
1. Train online TD3 for `segment_steps`.
2. Summarize `monitor.csv` to `run_summary.json`.
3. Ask LLM to propose a small JSON patch on HPRS constants.
4. Apply patch to produce `warehouse_llm_seg_xx.yaml`.
5. Validate old vs. new HPRS on the same model.
6. Accept or reject patch based on predefined rules.

Main orchestration script:
- `agent/tools/run_online_llm_loop.py`

## 4. Acceptance Policy
Current acceptance policy in `run_online_llm_loop.py`:
- `success_rate` must not decrease (within `accept_delta`),
- `collision_rate` must not increase (within `accept_collision_delta`),
- if success and collision are effectively unchanged, `mean_reward` must improve by at least `accept_reward_delta`.

For this run (`agent/logs/llm_runs/llm_loop_summary.json`):
- `accept_delta = 0.0`
- `accept_collision_delta = 0.0`
- `accept_reward_delta = 0.1`

## 5. Experimental Configuration
### 5.1 Pipeline
1. Expert collection (A* + DWA): `agent/expert/run_expert.py`
2. Offline pretraining (TD3-BC): `agent/td3/td3bc/train_td3bc.py`
3. Online baseline TD3: `agent/td3/online_td3/train_online_td3.py`
4. LLM online loop: `agent/tools/run_online_llm_loop.py`

### 5.2 Core configs
- Offline: `agent/config/offline_td3_bc.yaml`
- Online baseline: `agent/config/online_td3_baseline.yaml`
- Online LLM: `agent/config/online_td3_llm.yaml`
- Final comparison: `agent/config/evaluate_compare_models.yaml`

## 6. Quantitative Results
### 6.1 Final baseline vs LLM comparison
Source: `agent/logs/compare_models/compare_models.csv`

| Model | Success | Collision | Timeout | Mean Reward | Mean Success Reward | Mean Steps |
|---|---:|---:|---:|---:|---:|---:|
| `online_baseline_best` | 0.48 | 0.010 | 0.000 | -11.41 | 2.07 | 1362.98 |
| `online_llm_seg09_best` | 0.76 | 0.035 | 0.000 | -78.74 | 2.38 | 938.81 |

### 6.2 Final HPRS constant changes
Baseline HPRS: `auto-shaping/configs/warehouse.yaml`  
Final accepted LLM HPRS: `agent/logs/llm_runs/seg_09/accepted_hprs.yaml`

- `approach_dist`: `2.0 -> 0.96`
- `collision_penalty`: `1.0 -> 1.4`
- `delta_dist_weight`: `1.0 -> 1.2` (introduced earlier and retained)

## 7. Segment-by-Segment Patch Trace
Derived from direct YAML comparison (`warehouse_llm_seg_xx.yaml` vs the previous segment's `accepted_hprs.yaml`; for `seg_01`, compare against `auto-shaping/configs/warehouse.yaml`), plus `llm_patch.json` and `reject_reason.txt`.

| Segment | Compared against | Patch summary | Decision | Reason (if rejected) |
|---|---|---|---|---|
| `seg_01` | `auto-shaping/configs/warehouse.yaml` | `approach_dist: 2.0->1.6`, `delta_dist_weight: 1.0->1.2` | Accepted | - |
| `seg_02` | `seg_01/accepted_hprs.yaml` | No YAML parameter delta because `llm_patch.constants = {}`; candidate YAML equals previous accepted HPRS | Rejected | success_rate dropped by -0.100 |
| `seg_03` | `seg_02/accepted_hprs.yaml` | No YAML parameter delta because `llm_patch.constants = {}`; candidate YAML equals previous accepted HPRS | Rejected | success_rate dropped by -0.100 |
| `seg_04` | `seg_03/accepted_hprs.yaml` | No YAML parameter delta because `llm_patch.constants = {}`; candidate YAML equals previous accepted HPRS | Accepted | previous accepted config retained |
| `seg_05` | `seg_04/accepted_hprs.yaml` | `collision_penalty: 1.0->0.9` | Rejected | success_rate dropped by -0.100 |
| `seg_06` | `seg_05/accepted_hprs.yaml` | `approach_dist: 1.6->2.24`, `delta_dist_weight: 1.2->1.68`, `laser_safe: 1.5->0.9` | Rejected | success_rate dropped by -0.100 |
| `seg_07` | `seg_06/accepted_hprs.yaml` | `delta_dist_weight: 1.2->1.72` | Rejected | mean_reward did not improve (-0.42) |
| `seg_08` | `seg_07/accepted_hprs.yaml` | No YAML parameter delta because `llm_patch.constants = {}`; candidate YAML equals previous accepted HPRS | Accepted | previous accepted config retained |
| `seg_09` | `seg_08/accepted_hprs.yaml` | `approach_dist: 1.6->0.96`, `collision_penalty: 1.0->1.4` | Accepted | - |
| `seg_10` | `seg_09/accepted_hprs.yaml` | `delta_dist_weight: 1.2->1.24` (present in YAML delta even though `llm_patch.constants` is empty) | Rejected | success_rate dropped by -0.100 |

## 8. Figures
### 8.1 Final model comparison
![Final baseline vs LLM comparison](agent/logs/compare_models/compare_models_bars.png)

### 8.2 Offline checkpoint diagnostics
![Offline checkpoint comparison (bars)](agent/logs/compare_checkpoints_bars.png)

![Offline checkpoint comparison (lines)](agent/logs/compare_checkpoints_lines.png)

## 9. Reproduction Commands
### 9.1 Online baseline
```bash
python agent/td3/online_td3/train_online_td3.py \
  --mode warm_start \
  --config agent/config/online_td3_baseline.yaml
```

### 9.2 LLM-online loop
```bash
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

### 9.3 Final comparison
```bash
python agent/tools/compare_models_with_hprs.py \
  --config agent/config/evaluate_compare_models.yaml \
  --verbose
```

## 10. Limitations and Recommended Next Steps
Current findings show a clear success-rate gain but increased collision rate in final comparison. Recommended follow-up:
1. Tighten acceptance policy for safety-dominant deployment (`accept_collision_delta < 0` or explicit collision hard-threshold).
2. Add a risk-adjusted objective for patch acceptance (e.g., weighted score with collision penalty).
3. Evaluate with larger validation episode count to reduce variance before accepting patches.
4. Track confidence intervals over repeated seeds for publication-grade statistical claims.
