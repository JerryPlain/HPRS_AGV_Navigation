# Methodlogy

## 摘要
本文提出一条面向 AGV 仓储导航的 offline-to-online 强化学习方法链路。系统由四部分组成：经典规划专家数据生成（A* + DWA）、离线 TD3-BC 预训练、在线 TD3 微调、以及受约束的 LLM 外环奖励常数优化。整体目标是同时提高任务成功率与安全性，并保持过程可解释、可审计、可复现。

## 1. 问题建模
将导航任务定义为连续控制 MDP：
- 状态空间 `S`：机器人位姿/速度、目标相对几何量、可选激光扫描。
- 动作空间 `A`：底盘线速度和角速度连续控制。
- 转移 `P`：由 Unity 仿真动力学决定，通过 gRPC 桥接。
- 优化目标：在有限步内完成到达任务并尽量避免碰撞。

代码映射：
- `agent/envs/warehouse_unity_env.py`
- `agent/simulator_vec_env.py`
- `agent/envs/get_env.py`

## 2. 训练框架

### 2.1 专家数据构建
每个并行环境按如下过程采样：
1. 使用 A* 规划全局路径。
2. 使用 DWA/P 控制器生成局部动作。
3. 记录转移 `(s_t, a_t, r_t, s_{t+1}, d_t)`。
4. 仅保留成功回合进入离线数据集，提高样本质量。

代码映射：`agent/expert/run_expert.py`

### 2.2 离线 TD3-BC
Critic 目标：
`y_t = r_t + gamma * (1 - d_t) * min(Q1_target(s_{t+1}, a_tilde), Q2_target(s_{t+1}, a_tilde))`

目标动作平滑：
`a_tilde = clip(pi_target(s_{t+1}) + eps, -a_max, a_max)`，其中 `eps` 为裁剪高斯噪声。

Critic 损失：
`L_critic = MSE(Q1(s_t, a_t), y_t) + MSE(Q2(s_t, a_t), y_t)`

Actor 损失（延迟更新）由两项组成：
- TD3 价值项：最大化 `Q(s, pi(s))`
- BC 约束项：`||pi(s) - a||^2`

自适应系数：
`lambda = alpha_refined / mean(|Q(s, pi(s))|)`

最终：
`L_actor = -lambda * mean(Q(s, pi(s))) + MSE(pi(s), a)`

代码映射：
- `agent/td3/td3bc/td3_bc_agent.py`
- `agent/td3/td3bc/train_td3bc.py`

### 2.3 在线 TD3 微调
在线阶段从离线 checkpoint warm start actor/critic，并在仿真交互中持续更新 replay buffer 与网络参数，以适配在线分布。

代码映射：`agent/td3/online_td3/train_online_td3.py`

## 3. 自车中心观测与动作缩放
将世界坐标观测重参数化为自车中心特征：
- 目标距离：`d = sqrt(dx^2 + dy^2)`
- 相对方位：`bearing = atan2(dy, dx) - yaw`
- 导航向量：`[d_norm, sin(bearing), cos(bearing), v_norm, w_norm]`
- 若启用激光，则附加归一化 lidar 通道。

策略动作在 `[-1, 1]` 空间学习，执行前映射到物理控制范围：
`a_phys = ((a + 1) / 2) * (a_max - a_min) + a_min`

代码映射：`agent/td3/online_td3/egocentric_normalization_wrapper.py`

## 4. HPRS 层级势函数奖励塑形

### 4.1 层级语义
约束按优先级组织：
1. Safety（`ensure`）
2. Target（`achieve`/`conquer`）
3. Comfort（`encourage`）

约束由 YAML 定义并解析为可执行谓词。

代码映射：
- `auto-shaping/configs/warehouse.yaml`
- `auto-shaping/auto_shaping/spec/reward_spec.py`

### 4.2 谓词归一化评分
对变量 `v` 和区间 `[l, u]`，定义：
- `norm(v) = (clip(v, l, u) - l) / (u - l)`
- 对 `v <= tau`：`score = 1 - norm(v; [tau, u])`
- 对 `v >= tau`：`score = norm(v; [l, tau])`

这些评分用于计算安全、目标、舒适三层势函数。

### 4.3 势差奖励
基础稀疏奖励：
- 若成功标志为真，`r_base = 1`，否则为 `0`。

势差项：
- `Delta_phi_s = gamma * Phi_s(s') - Phi_s(s)`
- `Delta_phi_t = gamma * Phi_t(s') - Phi_t(s)`
- `Delta_phi_c = gamma * Phi_c(s') - Phi_c(s)`

核心塑形奖励：
- `r_core = r_base + shaping_scale * (Delta_phi_s + Delta_phi_t + Delta_phi_c)`

实现中还加入两项工程增强：
- 距离进度奖励：`w_delta_dist * (dist_prev - dist_now)`
- 碰撞惩罚：发生碰撞时减去 `w_collision`

最终奖励：
- `r = r_core + r_dist_bonus - r_collision_penalty`

说明：后两项是有意加入的非纯势函数项，用于强化收敛速度与安全约束。

代码映射：`auto-shaping/auto_shaping/hprs_vec_wrapper.py`

## 5. LLM 受约束外环优化

### 5.1 分段评估
每个训练 segment 结束后，从 `monitor.csv` 汇总：
- success rate
- collision rate
- mean reward
- mean success reward
- mean episode length

代码映射：`agent/tools/llm_summarize_run.py`

### 5.2 Patch 生成
LLM 仅可修改受限常数子集，且受幅度约束、参数白名单、最近变更规避、结构化 JSON 输出等规则控制。

代码映射：
- `agent/tools/llm_propose_patch.py`
- `agent/tools/run_llm_pipeline.py`
- `agent/tools/llm_apply_patch.py`

### 5.3 Patch 接受准则
新 patch 仅在以下条件满足时接受：
1. 成功率不劣于旧配置（在容忍阈值内）。
2. 碰撞率不劣于旧配置（在容忍阈值内）。
3. 若成功率和碰撞率均基本不变，则平均奖励必须提升到最小增量以上。

否则拒绝 patch，并保持上一版 HPRS 常数。

代码映射：`agent/tools/run_online_llm_loop.py`

## 6. 复现实验配置
核心配置文件：
- `agent/config/offline_td3_bc.yaml`
- `agent/config/online_td3_baseline.yaml`
- `agent/config/online_td3_llm.yaml`
- `agent/config/evaluate_compare_models.yaml`

## 7. 方法贡献
1. 提供完整的 offline-to-online AGV 导航训练范式。  
2. 在 VecEnv 层实现层级塑形，降低对仿真端侵入。  
3. 构建受约束、可审计、可回滚的 LLM 奖励常数优化外环。  
4. 通过显式接受准则将成功率与安全指标联合约束。  
