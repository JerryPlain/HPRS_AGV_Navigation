# ReGoRL 研究扩展草案（ICRA 2027）

## 1. 定位

**ReGoRL: Reward Governance for Safety-Critical Robot Learning via Hierarchy-Guided Patches**

本文拟将当前仓库中的 `VTPRL` 前身工作，从“LLM 辅助 HPRS 常数调参”推进为一个更完整的研究问题：

- 不是 reward generation
- 不是 unrestricted reward search
- 而是 **training-time reward governance over hierarchical reward specifications**

核心思想是：在安全关键机器人强化学习中，奖励通常已经存在人为构建的层级化规范，真正困难的不是“从零写 reward”，而是在训练过程中 **如何安全、局部、可验证地更新 reward specification**。

## 2. 与现有工作的边界

### 2.1 与 TEXT2REWARD 的区别

TEXT2REWARD 的核心是从自然语言目标和环境抽象生成可解释的 dense reward code，并可结合 human feedback 持续 refine。

ReGoRL 不做从零生成 reward code。我们假设 reward 已经以结构化层级规范存在，研究重点是：

- 如何定义可编辑 patch space
- 如何验证 patch 的语法、结构与层级合法性
- 如何在训练中治理 patch 的接受、拒绝与回滚

### 2.2 与 EUREKA 的区别

EUREKA 的核心是让 LLM 结合环境代码生成可执行 reward code，并通过 evolutionary search 与 reflection 优化 reward candidates。

ReGoRL 不在整个 reward program space 中自由搜索。我们只在 **hierarchy-guided local patch space** 中工作，强调：

- editable vs protected regions
- bounded local updates
- hierarchy preservation
- safety-consistent selection
- rollback-safe deployment

### 2.3 与 HPRS 的关系

HPRS 提供的是一种将任务表示为 `safety > target > comfort` 偏序结构，并自动构造 hierarchical potential-based reward 的方法。

ReGoRL 建立在 HPRS 之上，但贡献不在于重新定义 HPRS，而在于将 HPRS 的层级语义进一步提升为：

- patch legality prior
- verifier rule source
- selector ranking principle
- governance objective backbone

也就是说，HPRS 提供 reward semantics，ReGoRL 负责 reward update governance。

### 2.4 与 VTPRL precursor 的关系

当前 `VTPRL` 已经证明：在 HPRS 上做受约束的 segment-wise reward 常数调整，可以提高 success 和效率，但 collision 也会上升。这说明单一 proposer 驱动的外环调参会出现明显的 **safety-efficiency drift**。

因此，ReGoRL 的出发点不是让 proposer 更激进，而是把 reward adaptation 从单点 proposal 问题，升级为 **multi-agent governed decision problem**。

## 3. 论文核心问题表述

### 3.1 Problem Formulation

我们考虑安全关键机器人强化学习中的训练时奖励更新问题。设当前奖励规范为 `S_t ∈ 𝒮`，它不是任意 reward code，而是一个带有层级与约束的结构化 specification，包含：

- hierarchy: `safety > target > comfort`
- gating dependencies
- adjustable parameters: weights, thresholds, margins
- bounded template clauses
- protected constraints: mandatory safety terms, module topology, hierarchy invariants, hard parameter bounds

在每次 reward adaptation step，系统观察近期训练诊断 `z_t`，包括：

- success rate
- collision rate
- timeout rate
- mean episode length
- mean reward
- rollout failure summary
- accepted/rejected patch history

系统随后从约束 patch space `𝒫(S_t)` 中生成候选 patch，并输出治理动作：

`a_t ∈ {accept p_t*, reject all and keep S_t}`

目标是在 **不破坏层级结构与安全优先级** 的前提下，提升策略表现。如果没有足够安全且有效的 patch，则保持当前 specification 不变。

### 3.2 Reward Patch

定义 patch `p_t : S_t → S'_t` 为对现有 reward specification 的局部修改，而不是全量重写。

这一定义的关键在于：

- patch operates on specification, not free-form code
- patch is local and auditable
- patch can be rejected or rolled back without causing uncontrolled semantic drift

### 3.3 Patch Validity

一个 patch 被认为合法，需要满足三层条件：

1. Executable validity  
   patch 可解析、字段存在、类型匹配、参数范围有效、配置可加载。

2. Structural validity  
   patch 不得修改 protected fields，不得破坏 HPRS 模块拓扑，不得删除 gate dependency，不得改变 hierarchy order。

3. Governance validity  
   patch 不得造成不可接受的 safety regression，不得允许低优先级模块支配高优先级语义，且 patch scope 与 magnitude 需要与近期诊断相匹配。

## 4. 方法总览

### 4.1 High-Level Pipeline

```text
Reward Specification S_t
        ↓
Patch Space Definition 𝒫(S_t)
        ↓
Patch Proposer (LLM)
        ↓
Verifier (syntax / structure)
        ↓
Safety Critic
        ↓
Hierarchy-Aware Selector
        ↓
Accept / Reject / Rollback
        ↓
Updated Specification S_{t+1}
        ↓
RL Training Continues
```

### 4.2 Figure 1: Hierarchy-Guided Reward Patch Governance

```text
                           ROBOTIC RL TRAINING LOOP
┌───────────────────────────────────────────────────────────────────────┐
│ Policy πθ interacts with environment                                 │
│ Observation → Action → Transition → Replay Buffer → TD3 Update      │
│ Training diagnostics z_t                                             │
│ (success, collision, timeout, rollout failure logs, patch history)  │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
                    REWARD GOVERNANCE TRIGGER
              (stagnation / safety degradation / instability)

                     HIERARCHY-GUIDED PATCH GOVERNANCE
┌───────────────────────────────────────────────────────────────────────┐
│ Patch Proposer (LLM)                                                 │
│   uses z_t + current reward spec S_t                                 │
│                         │                                             │
│                         ▼                                             │
│              Candidate Patches {p1, ..., pk}                         │
│                         │                                             │
│                         ▼                                             │
│                Verifier: syntax + structural validity                │
│                         │                                             │
│                         ▼                                             │
│                Safety Critic: unsafe trade-off detection             │
│                         │                                             │
│                         ▼                                             │
│                Selector: lexicographic hierarchy-aware ranking       │
│                priority = validity → safety → success → efficiency   │
│                         │                                             │
│                         ▼                                             │
│                   ACCEPT PATCH / REJECT / ROLLBACK                   │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
                    UPDATED REWARD SPECIFICATION
                           S_t → S_{t+1}
                                │
                                ▼
                        CONTINUE RL TRAINING
```

## 5. Patch Space 设计

### 5.1 Why Patch Space Instead of Full Rewrite

与 EUREKA 一类的自由 reward code search 相比，面向安全关键机器人 RL 的 unrestricted search space 过大，容易带来：

- execution bugs
- semantic drift
- hierarchy disruption
- weak auditability
- rollback difficulty

因此 ReGoRL 采用 **hierarchy-guided local patch space**。它的优势是：

- search space 更小
- 静态验证更容易
- 可定义权限边界
- 更适合训练过程中的迭代 refinement
- 更容易保留已有人类知识和任务结构

### 5.2 Patch Taxonomy

```text
Parameter Patch
  change scalar constants

Gate Patch
  adjust activation conditions without altering hierarchy dependency

Template Patch
  add/remove bounded local shaping clauses within predefined templates
```

### 5.3 Figure 2: Hierarchical Reward Patch Space

```text
                HIERARCHICAL REWARD SPECIFICATION

                 ┌─────────────────────────────┐
                 │         SAFETY MODULE       │
                 │ Protected:                  │
                 │ • safety dominance          │
                 │ • collision penalty         │
                 │ Editable:                   │
                 │ • penalty scale             │
                 │ • safety margin             │
                 │ Allowed patches:            │
                 │ • parameter patch           │
                 │ • conservative gate patch   │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │         TARGET MODULE       │
                 │ Protected:                  │
                 │ • safety gate dependency    │
                 │ • terminal success signal   │
                 │ Editable:                   │
                 │ • progress weight           │
                 │ • goal threshold            │
                 │ Allowed patches:            │
                 │ • parameter patch           │
                 │ • gate patch                │
                 │ • template patch            │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │        COMFORT MODULE       │
                 │ Protected:                  │
                 │ • cannot bypass hierarchy   │
                 │ Editable:                   │
                 │ • smoothness weight         │
                 │ • oscillation penalty       │
                 │ Allowed patches:            │
                 │ • parameter patch           │
                 │ • gate patch                │
                 │ • template patch            │
                 └─────────────────────────────┘
```

### 5.4 Editable vs Protected Regions

**Protected region**

- hierarchy order: `safety > target > comfort`
- mandatory safety gate dependency
- existence of collision penalty / terminal success signal
- core HPRS module topology
- hard parameter range limits for critical safety terms

**Editable region**

- reward weights
- threshold values
- soft gate parameters
- activation margins
- selected local shaping clauses from a bounded template library

### 5.5 Hierarchy-Guided Constraints

Rule 1: Priority-dependent edit freedom  
高优先级模块可编辑自由度更低。

- safety layer: only conservative edits
- target layer: moderate edits
- comfort layer: broader edits

Rule 2: Upward non-domination  
低优先级 patch 不得改变高优先级语义。

- comfort patch 不能让 unsafe behavior 变得高回报
- target patch 不能绕过 safety gate
- target/comfort patch 不能通过重加权淹没 safety penalty

Rule 3: Locality bias  
优先小而局部的 patch，因为它们更容易：

- verify
- audit
- attribute
- rollback

## 6. 多代理治理框架

### 6.1 Agent Roles

**Patch Proposer**

- 输入：`S_t`, `z_t`, patch history
- 输出：候选 patch 集合 `{p_1, ..., p_k}`
- 责任：提出与近期训练症状一致的局部修改假设

**Verifier**

- 责任：检查 executable validity 与 structural validity
- 输出：invalid / structurally valid 标签与失败类型

**Safety Critic**

- 责任：判断 patch 是否存在明显的 safety trade-off 风险
- 典型规则：
  - no explicit weakening of protected safety constraints
  - no predicted success-via-risk shortcut
  - no excessive safety-layer edit magnitude

**Hierarchy-Aware Selector**

- 只在 verifier 和 critic 过滤后的候选中选择
- 采用 lexicographic ranking：
  - validity first
  - safety second
  - success third
  - efficiency / comfort fourth
  - simplicity / locality fifth

### 6.2 Verifier Rules

| Rule ID | Level | Rule | Failure Type | Action |
|---|---|---|---|---|
| R1 | Executable | field path must exist | invalid field | reject |
| R2 | Executable | value type must match schema | type mismatch | reject |
| R3 | Executable | parameter must stay within allowed bounds | out-of-range | reject |
| R4 | Structural | protected fields cannot be modified | protected edit | reject |
| R5 | Structural | target must remain gated by safety | hierarchy violation | reject |
| R6 | Structural | comfort cannot activate without higher-level conditions | hierarchy violation | reject |
| R7 | Governance | safety-layer edits must stay conservative | excessive safety edit | reject or down-rank |
| R8 | Governance | low-priority patch cannot reduce effective safety dominance | unsafe trade-off | reject |
| R9 | Governance | patch must be justified by recent diagnostics | unjustified patch | down-rank |
| R10 | Governance | prefer smaller local patches when gains are similar | unnecessary complexity | down-rank |

## 7. 数学化表述建议

### 7.1 Candidate Filtering

对候选 patch `p ∈ 𝒫(S_t)`，定义：

- `V_exec(p, S_t) ∈ {0,1}`: executable validity
- `V_struct(p, S_t) ∈ {0,1}`: structural validity
- `V_safe(p, z_t, S_t) ∈ {0,1}`: safety-consistency validity

只有满足

`V_exec(p, S_t) = 1, V_struct(p, S_t) = 1, V_safe(p, z_t, S_t) = 1`

的候选，才进入选择阶段。

### 7.2 Governance Objective

可以将治理目标写为受约束优化：

`p_t* = argmax_{p ∈ 𝒫(S_t)} U(p ; z_t, S_t)`

subject to

- `V_exec(p, S_t) = 1`
- `V_struct(p, S_t) = 1`
- `V_safe(p, z_t, S_t) = 1`

其中 `U` 不是单一标量 reward，而是一个符合层级语义的 lexicographic utility：

`validity ≻ safety ≻ success ≻ efficiency ≻ simplicity`

如果没有候选满足约束，则：

`S_{t+1} = S_t`

### 7.3 Accept / Reject Decision

给定候选集 `C_t`，系统输出：

- `accept(p_t*)` if there exists a candidate satisfying governance constraints and sufficient utility gain
- `reject_all` otherwise

这一定义突出了 ReGoRL 与 reward generation 的根本区别：系统的核心输出不是 reward code，而是 **governed update decision**。

## 8. 实验设计

### 8.1 需要补强的三项

你已经明确指出目前必须补的三项，这里把它们转成论文实验设计：

1. `multi-tasks`
- navigation
- manipulation
- harder safety-critical variants

2. `strong baselines + ablations`
- 固定 HPRS
- PSO / black-box parameter search
- single-agent patch proposer
- multi-agent without hierarchy guidance
- full ReGoRL

3. `method figure + formulation`
- Figure 1: governance pipeline
- Figure 2: patch space
- Sec. Problem Formulation
- Sec. Patch Space
- Sec. Multi-Agent Governance

### 8.2 Main Research Questions

**Q1. Does hierarchy-guided reward governance improve task outcomes over baseline reward tuning?**

Baselines:

- fixed HPRS
- PSO
- single-agent patch proposer
- multi-agent without hierarchy guidance
- full ReGoRL

**Q2. Does the framework reduce unsafe or invalid reward updates?**

Metrics:

- unsafe patch proposal rate
- unsafe patch filtered rate
- invalid patch rejection rate
- accepted patch precision

**Q3. What is the contribution of hierarchy-aware constraints and each agent role?**

Ablations:

- no safety critic
- no verifier
- no hierarchy-guided patch bounds
- no candidate ranking
- single candidate vs multiple candidates

**Q4. Does the method remain robust in harder or shifted safety-critical settings?**

Stress settings:

- denser obstacles
- tighter safety margins
- shifted starts / goals
- different warehouse layouts
- manipulation tasks
- sim-to-real transfer

### 8.3 Evaluation Metrics

**Policy-level**

- success rate
- collision rate
- timeout rate
- mean episode length
- mean reward

**Reward-update reliability**

- invalid patch rate
- unsafe patch proposal rate
- unsafe patch filtered rate
- accepted patch rate
- accepted patch improvement rate
- rollback frequency
- cumulative gain per accepted patch

**Governance-level**

- hierarchy violation count
- average patch size
- average edit locality
- selector disagreement rate
- verifier-caught error categories

## 9. 多任务与泛化

### 9.1 Navigation Task

当前仓库天然支撑的主实验场景：

- AGV warehouse navigation
- offline-to-online TD3 training
- HPRS reward initialization

这是 ReGoRL 的主验证基线，也是与 `VTPRL` 连续性最强的部分。

### 9.2 Manipulation Task

为证明方法不是导航特化规则，第二类任务应切换到 manipulation benchmark。关键不是复用相同 reward 项，而是复用相同治理思想：

- safety module: contact safety, joint limit, collision avoidance
- target module: grasp / place / alignment success
- comfort module: smoothness, force regularity, efficiency

这能突出 ReGoRL 的核心主张：它治理的是 **hierarchical reward update process**，而不是某个单独的导航 reward trick。

## 10. 论文结构建议

### 10.1 Introduction

建议三段式开场：

1. 在安全关键机器人 RL 中，reward design 不是一次性 coding，而是持续性的 engineering problem。
2. 现有工作关注 reward generation 或 reward search，但没有解决 governed reward revision。
3. 我们将问题重述为 **hierarchical reward patch engineering**，并提出 ReGoRL。

### 10.2 Related Work

- LLM-assisted reward generation/search
- structured reward shaping / HPRS
- safe RL
- multi-agent scientific / engineering workflows

### 10.3 Problem Formulation

定义：

- hierarchical reward specification
- patch
- patch validity
- governance objective
- accept/reject decision

### 10.4 Method

- Hierarchy-Guided Reward Patch Space
- Verifier and Safety Critic
- Hierarchy-Aware Selector
- Rollback and update policy

### 10.5 Experiments

- AGV warehouse navigation
- manipulation benchmark
- stronger safety-shift settings
- baselines and ablations

### 10.6 Discussion

主线建议明确写成一句话：

> In safety-critical robotic reinforcement learning, the most suitable role for large language models is not that of an unconstrained reward generator, but that of a structured reward governor.

## 11. 核心贡献写法

建议保留为四点：

1. We reformulate training-time LLM-assisted reward tuning as a hierarchical reward patch engineering problem over structured reward specifications, moving beyond reward generation and unrestricted reward search.

2. We introduce a hierarchy-guided reward patch space with editable regions, protected constraints, and bounded local updates, enabling auditable and rollback-safe reward adaptation.

3. We propose a multi-agent reward governance framework that decomposes reward adaptation into patch proposal, safety critique, structural verification, and governed candidate selection.

4. We introduce reward-update reliability as a first-class evaluation target for LLM-based reward engineering, including invalid-patch rejection, unsafe-patch filtering, and accepted-patch consistency, in addition to policy-level performance.

## 12. 与当前仓库实现的映射关系

当前代码库中已经存在的对应基础：

- `HPRS` reward specification and wrapper
- segment-wise LLM patch proposal
- apply / accept / reject loop
- rollback-style persistence of accepted configs
- navigation offline-to-online RL pipeline

仍需补齐的关键差距：

- 从“单 patch JSON 常数调整”升级为正式 patch space
- 从“单一 acceptance rule”升级为 verifier + critic + selector
- 从“单任务导航”扩展到 multi-task evaluation
- 从“policy metrics”扩展到 reward-update reliability metrics
- 从“经验性 README 报告”升级为完整 paper formulation

## 13. 下一步落地建议

按照最小可执行路径，建议下一阶段按下面顺序推进：

1. 先把 `patch grammar / editable-protected schema / verifier rules` 固化成配置接口。
2. 把当前 `run_online_llm_loop.py` 拆成 proposer, verifier, critic, selector 四步。
3. 在导航任务中先做 parameter patch + gate patch，暂不急于开放 template patch。
4. 建立 reward-update reliability 日志格式，确保 unsafe / invalid / rollback 能被直接统计。
5. 在第二任务中验证方法迁移性，优先选择一个 reward hierarchy 明确的 manipulation setting。

## 14. 一句话总结

ReGoRL 的核心不是让 LLM 更自由地写 reward，而是让 LLM 在一个 **层级约束、可验证、可回滚的 patch governance framework** 中，安全地参与 reward adaptation。
