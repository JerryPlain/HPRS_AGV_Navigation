# Publication-Ready Methodology (中文)

## 摘要
本文提出一条用于 AGV 仓储导航的离线到在线强化学习方法链路：以经典规划器生成专家数据，使用 TD3-BC 进行离线策略初始化，再通过在线 TD3 适配交互分布，并在外环引入基于大语言模型（LLM）的层级势函数奖励塑形（HPRS）常数调优。该设计在保证可解释性与可审计性的前提下，兼顾任务成功率与安全指标。

## 1. 问题定义
将仓储导航建模为连续控制 MDP：
$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma).
$$
状态由 Unity 环境返回（机器人位姿/速度、目标相对量、激光扫描），动作为底盘线速度与角速度。目标是在有限步长内到达目标位姿，同时降低碰撞风险。

实现映射：
- 环境：`agent/envs/warehouse_unity_env.py`
- 向量化通信：`agent/simulator_vec_env.py`
- 训练环境组装：`agent/envs/get_env.py`

## 2. 分阶段训练框架

### 2.1 专家数据采集（A* + DWA）
在每个并行环境中，先规划全局路径，再由局部控制器输出动作并采集转移：
$$
\mathcal{D}=\{(s_t,a_t,r_t,s_{t+1},d_t)\}_{t=1}^{N}.
$$
工程上仅保留成功回合样本以提高离线数据质量。

实现映射：`agent/expert/run_expert.py`

### 2.2 离线 TD3-BC 预训练
离线阶段使用双 Q 目标（TD3）+ 行为克隆约束（BC）。

目标 Q：
$$
y_t = r_t + \gamma (1-d_t)\min_{i=1,2}Q_{\bar\phi_i}(s_{t+1}, \tilde a_{t+1}),
$$
其中
$$
\tilde a_{t+1}=\text{clip}\left(\pi_{\bar\theta}(s_{t+1})+\epsilon,\,-a_{\max},a_{\max}\right),\quad
\epsilon\sim\text{clip}(\mathcal{N}(0,0.05^2),-0.2,0.2).
$$

Critic 损失：
$$
\mathcal{L}_{critic}=\sum_{i=1}^{2}\left\|Q_{\phi_i}(s_t,a_t)-y_t\right\|_2^2.
$$

Actor 损失（策略延迟更新）：
$$
\lambda = \frac{\alpha_{\text{refined}}}{\mathbb{E}_{s\sim\mathcal D}[|Q_{\phi_1}(s,\pi_\theta(s))|]},
$$
$$
\mathcal{L}_{actor}= -\lambda\cdot \mathbb{E}[Q_{\phi_1}(s,\pi_\theta(s))]
\;+\;
\mathbb{E}\left[\|\pi_\theta(s)-a\|_2^2\right].
$$
其中 $\alpha_{\text{refined}}=\alpha\cdot\text{policy\_refinement\_factor}$。

实现映射：`agent/td3/td3bc/td3_bc_agent.py`、`agent/td3/td3bc/train_td3bc.py`

### 2.3 在线 TD3 微调（Warm Start）
在线阶段从离线 checkpoint 加载 actor/critic 权重，并在环境交互中持续更新 replay buffer 与网络参数，实现离线到在线迁移。

实现映射：`agent/td3/online_td3/train_online_td3.py`

## 3. 观测与动作归一化（Egocentric）
为稳定训练，观测从世界坐标重参数化为自车中心极坐标特征：
$$
d=\sqrt{\Delta x^2+\Delta y^2},\quad
\beta=\text{atan2}(\Delta y,\Delta x)-\psi.
$$
导航主特征为
$$
\left[\frac{d}{d_{\max}},\sin\beta,\cos\beta,\frac{v}{v_{\max}},\frac{\omega}{\omega_{\max}}\right],
$$
激光分量线性缩放到 $[0,1]$。

策略输出动作 $a\in[-1,1]^2$，执行前反归一化：
$$
a_{\text{phys}}=\frac{a+1}{2}\odot(a_{\max}-a_{\min})+a_{\min}.
$$

实现映射：`agent/td3/online_td3/egocentric_normalization_wrapper.py`

## 4. HPRS：层级势函数奖励塑形

### 4.1 层级语义
奖励约束按优先级组织：
1. Safety (`ensure`)
2. Target (`achieve`/`conquer`)
3. Comfort (`encourage`)

规格由 YAML 定义并解析为可执行规则。  
实现映射：`auto-shaping/configs/warehouse.yaml`、`auto-shaping/auto_shaping/spec/reward_spec.py`

### 4.2 归一化谓词奖励
定义
$$
\text{norm}(v; l,u)=\frac{\text{clip}(v,l,u)-l}{u-l}\in[0,1].
$$
对于约束 $v\le \tau$：
$$
\rho(v;\tau)=1-\text{norm}(v;\tau,u),
$$
对于约束 $v\ge \tau$：
$$
\rho(v;\tau)=\text{norm}(v;l,\tau).
$$

其中 $(l,u)$ 来自变量边界配置。

### 4.3 三层势函数
安全掩码：
$$
m_s(s)=\prod_{k\in\mathcal{S}}\mathbf{1}\left[g_k(s)\right].
$$

Target 势：
$$
\Phi_t(s)=m_s(s)\cdot\frac{1}{|\mathcal{T}|}\sum_{k\in\mathcal{T}}\rho_k(s).
$$

Safety 势（代码中为 ensure 命中数）：
$$
\Phi_s(s)=\sum_{k\in\mathcal{S}}\mathbf{1}\left[g_k(s)\right].
$$

Comfort 势（软门控）：
$$
m_t(s)=\frac{1}{|\mathcal{T}|}\sum_{k\in\mathcal{T}}\rho_k(s),\quad
\Phi_c(s)=m_s(s)\,m_t(s)\sum_{j\in\mathcal{C}}\rho_j(s).
$$

### 4.4 最终奖励
基础成功奖励：
$$
r_{\text{base}}(s')=\mathbf{1}[\text{success}(s')].
$$

势差塑形（非终止且有前态时）：
$$
r_{\text{pot}} = r_{\text{base}}
\lambda_{\text{shape}}\Big(
\gamma\Phi_s(s')-\Phi_s(s)
\gamma\Phi_t(s')-\Phi_t(s)
\gamma\Phi_c(s')-\Phi_c(s)
\Big).
$$

代码还引入两项附加密集项：
$$
r_{\Delta d}=w_{\Delta d}\cdot(d_{t}-d_{t+1}),\qquad
r_{\text{col}}=-w_{\text{col}}\cdot\mathbf{1}[\text{collision}].
$$

最终：
$$
r_t = r_{\text{pot}} + r_{\Delta d} + r_{\text{col}}.
$$

注：后两项不是标准 potential-based 项，会改变最优策略不变性；此处是有意的工程增强，用于加速收敛和强化安全惩罚。

实现映射：`auto-shaping/auto_shaping/hprs_vec_wrapper.py`

## 5. LLM 外环常数优化（受约束的策略外调参）

### 5.1 指标汇总
在每个训练段（segment）结束后，从 `monitor.csv` 统计：
$$
\text{success\_rate},\;
\text{collision\_rate},\;
\text{mean\_reward},\;
\text{mean\_success\_reward},\;
\text{mean\_length}.
$$
实现映射：`agent/tools/llm_summarize_run.py`

### 5.2 Patch 生成与约束
LLM 仅允许修改常数子集，且每轮限制小幅更新（含参数白名单、幅度约束、最近改动规避）。

实现映射：`agent/tools/llm_propose_patch.py`、`agent/tools/run_llm_pipeline.py`、`agent/tools/llm_apply_patch.py`

### 5.3 接受准则
给定旧/新指标 $(sr_o,cr_o,r_o)$、$(sr_n,cr_n,r_n)$ 与容忍度 $(\delta_s,\delta_c,\delta_r)$：
$$
\text{success\_not\_worse}: sr_n \ge sr_o-\delta_s,
$$
$$
\text{collision\_not\_worse}: cr_n \le cr_o+\delta_c.
$$
若成功率与碰撞率近似不变（在容忍度内），则要求奖励提升：
$$
|sr_n-sr_o|\le \delta_s\;\land\;|cr_n-cr_o|\le \delta_c
\Rightarrow r_n \ge r_o+\delta_r.
$$
满足以上条件才接受 patch，否则拒绝并保留旧 HPRS。

实现映射：`agent/tools/run_online_llm_loop.py`

## 6. 可复现实验配置（代码基线）
- 离线训练配置：`agent/config/offline_td3_bc.yaml`
- 在线 baseline：`agent/config/online_td3_baseline.yaml`
- 在线 LLM-HPRS：`agent/config/online_td3_llm.yaml`
- 对比评估：`agent/config/evaluate_compare_models.yaml`

## 7. 贡献总结（方法层）
1. 提出可工程落地的 offline-to-online AGV 导航训练范式。  
2. 在 VecEnv 层实现层级奖励塑形，保持仿真端低侵入。  
3. 将 LLM 限定在“受约束、可验证、可回滚”的外环奖励常数优化中。  
4. 给出面向安全与成功率的显式 patch 接受机制。  
