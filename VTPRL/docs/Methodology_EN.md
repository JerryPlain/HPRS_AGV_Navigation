# Methodlogy

## Abstract
This work presents an offline-to-online reinforcement learning pipeline for AGV warehouse navigation. The system combines classical expert trajectory generation (A* + DWA), offline TD3-BC pretraining, online TD3 finetuning, and a constrained LLM outer loop for hierarchical reward-shaping constant updates. The method is designed for practical deployment: traceable, safety-aware, and reproducible.

## 1. Problem Setup
We model warehouse navigation as a continuous-control MDP:
- State space `S`: robot pose/velocity, goal-relative geometry, and optional lidar scan.
- Action space `A`: continuous linear and angular velocity commands.
- Transition `P`: Unity simulator dynamics via gRPC bridge.
- Objective: maximize task completion performance while controlling collision risk.

Code mapping:
- `agent/envs/warehouse_unity_env.py`
- `agent/simulator_vec_env.py`
- `agent/envs/get_env.py`

## 2. Training Architecture

### 2.1 Expert Dataset Construction
For each parallel environment:
1. Plan a global route with A*.
2. Generate local control using DWA/P-controller.
3. Record transitions `(s_t, a_t, r_t, s_{t+1}, d_t)`.
4. Keep successful episodes to build a high-quality offline dataset.

Code mapping: `agent/expert/run_expert.py`

### 2.2 Offline TD3-BC
Critic target:
`y_t = r_t + gamma * (1 - d_t) * min(Q1_target(s_{t+1}, a_tilde), Q2_target(s_{t+1}, a_tilde))`

Target action smoothing:
`a_tilde = clip(pi_target(s_{t+1}) + eps, -a_max, a_max)`, with clipped Gaussian noise.

Critic loss:
`L_critic = MSE(Q1(s_t, a_t), y_t) + MSE(Q2(s_t, a_t), y_t)`

Actor loss (delayed update):
- TD3 objective term: maximize `Q(s, pi(s))`
- BC regularization term: `||pi(s) - a||^2`
- Adaptive scaling:
`lambda = alpha_refined / mean(|Q(s, pi(s))|)`

Final actor objective:
`L_actor = -lambda * mean(Q(s, pi(s))) + MSE(pi(s), a)`

Code mapping:
- `agent/td3/td3bc/td3_bc_agent.py`
- `agent/td3/td3bc/train_td3bc.py`

### 2.3 Online TD3 Finetuning
The online stage warm-starts actor and critic from offline checkpoints, then continues interaction-driven updates in the simulator. Replay buffer is continuously refreshed with online trajectories.

Code mapping: `agent/td3/online_td3/train_online_td3.py`

## 3. Egocentric Observation and Action Scaling
Raw world-frame features are transformed into an egocentric representation:
- Distance to goal: `d = sqrt(dx^2 + dy^2)`
- Relative bearing: `bearing = atan2(dy, dx) - yaw`
- Navigation features: `[d_norm, sin(bearing), cos(bearing), v_norm, w_norm]`
- Optional lidar channels are min-max scaled.

Action is trained in normalized space `[-1, 1]` and mapped to physical limits before stepping:
`a_phys = ((a + 1) / 2) * (a_max - a_min) + a_min`

Code mapping: `agent/td3/online_td3/egocentric_normalization_wrapper.py`

## 4. Hierarchical Potential-Based Reward Shaping (HPRS)

### 4.1 Hierarchy
Constraints are structured by priority:
1. Safety (`ensure`)
2. Target (`achieve`, `conquer`)
3. Comfort (`encourage`)

The hierarchy is specified in YAML and parsed into executable predicates.

Code mapping:
- `auto-shaping/configs/warehouse.yaml`
- `auto-shaping/auto_shaping/spec/reward_spec.py`

### 4.2 Normalized Predicate Score
For variable value `v` and bounds `[l, u]`:
- `norm(v) = (clip(v, l, u) - l) / (u - l)`
- For `v <= tau`: score = `1 - norm(v; [tau, u])`
- For `v >= tau`: score = `norm(v; [l, tau])`

These scores are used to compute hierarchical shaping potentials.

### 4.3 Potentials and Shaped Reward
Base sparse reward:
- `r_base = 1` if success flag is true, else `0`.

Potential-difference shaping:
- `Delta_phi_s = gamma * Phi_s(s') - Phi_s(s)`
- `Delta_phi_t = gamma * Phi_t(s') - Phi_t(s)`
- `Delta_phi_c = gamma * Phi_c(s') - Phi_c(s)`

Core shaped reward:
- `r_core = r_base + shaping_scale * (Delta_phi_s + Delta_phi_t + Delta_phi_c)`

Implementation-specific dense terms:
- Distance-progress bonus: `w_delta_dist * (dist_prev - dist_now)`
- Collision penalty: `-w_collision` when collision flag is active

Final reward:
- `r = r_core + r_dist_bonus - r_collision_penalty`

Note: the extra distance and collision terms are deliberate engineering additions beyond pure potential shaping.

Code mapping: `auto-shaping/auto_shaping/hprs_vec_wrapper.py`

## 5. Constrained LLM Outer Loop

### 5.1 Segment Evaluation
After each online segment, summarize:
- success rate
- collision rate
- mean reward
- mean success reward
- mean episode length

Code mapping: `agent/tools/llm_summarize_run.py`

### 5.2 Patch Generation
LLM proposes updates to a restricted constant set, under strict constraints (allowed params, bounded change ratio, recent-change avoidance, structured JSON output).

Code mapping:
- `agent/tools/llm_propose_patch.py`
- `agent/tools/run_llm_pipeline.py`
- `agent/tools/llm_apply_patch.py`

### 5.3 Acceptance Rule
A new patch is accepted only if:
1. Success rate is not worse than old result within tolerance.
2. Collision rate is not worse than old result within tolerance.
3. If success and collision are unchanged, mean reward must improve by a minimum margin.

Otherwise, the patch is rejected and the previous HPRS configuration remains active.

Code mapping: `agent/tools/run_online_llm_loop.py`

## 6. Reproducibility
Primary experiment configs:
- `agent/config/offline_td3_bc.yaml`
- `agent/config/online_td3_baseline.yaml`
- `agent/config/online_td3_llm.yaml`
- `agent/config/evaluate_compare_models.yaml`

## 7. Technical Contributions
1. A complete offline-to-online AGV RL training workflow with practical warm-start transfer.
2. VecEnv-level hierarchical shaping that is simulator-decoupled and auditable.
3. A constrained LLM reward-engineering loop with explicit safety-aware acceptance.
4. Traceable implementation path from configuration to evaluation artifacts.
