# VTPRL 仓库详解（Repo Guide）

本文是面向开发与复现实验的仓库导读，目标是让你快速看清：
- 项目在做什么
- 代码入口在哪里
- 训练、评估、LLM 外环如何串起来
- 日志和产物落在哪

## 1. 项目在做什么

该仓库实现的是一条 AGV 仓储导航的强化学习流水线：
1. 经典方法（A* + DWA）采集专家轨迹
2. 离线 TD3-BC 预训练（利用专家数据）
3. 在线 TD3 微调（与 Unity 仿真交互）
4. 可选：LLM 作为“外环”只调整 HPRS 奖励常数，并通过验收规则决定是否接受 patch

一句话：**这是一个 offline-to-online 的 AGV 导航训练框架，附带可审计的 LLM 奖励调参闭环。**

## 2. 仓库总览树（核心视角）

> 说明：下列树是“可理解结构”，省略了大体积二进制文件和第三方库内部细节。

```text
VTPRL/
├── README.md                         # 项目主说明（运行流程与命令）
├── requirements_llm.txt              # LLM 外环依赖
├── docs/
│   ├── Methodology_CN.md             # 方法论（中文）
│   ├── Methodology_EN.md             # 方法论（英文）
│   ├── ReGoRL_Paper_Outline_CN.md    # ReGoRL 论文扩展草案
│   ├── ReGoRL_Project_Structure_EN.md # ReGoRL 英文结构说明与 Mermaid 图
│   ├── Configuration-Parameters.md   # 参数说明
│   └── Repo_Guide_CN.md              # 本文档
├── agent/                            # 训练与评估主代码
│   ├── config/                       # 实验配置 YAML
│   │   ├── expert_collection.yaml
│   │   ├── offline_td3_bc.yaml
│   │   ├── online_td3_baseline.yaml
│   │   ├── online_td3_llm.yaml
│   │   ├── evaluate_agent.yaml
│   │   └── evaluate_compare_models.yaml
│   ├── envs/                         # 环境封装（Unity -> Gym 接口）
│   │   ├── get_env.py
│   │   ├── warehouse_unity_env.py
│   │   └── entities/warehouse_observation_entity.py
│   ├── expert/                       # 专家数据采集（A* + DWA）
│   │   ├── run_expert.py
│   │   └── expert_dataset.py
│   ├── td3/
│   │   ├── td3_agent.py              # TD3 基础实现
│   │   ├── utils/replay_buffer.py
│   │   ├── td3bc/
│   │   │   ├── td3_bc_agent.py       # TD3-BC 算法实现
│   │   │   └── train_td3bc.py        # 离线训练入口
│   │   └── online_td3/
│   │       ├── train_online_td3.py   # 在线训练入口
│   │       └── egocentric_normalization_wrapper.py
│   ├── tools/                        # 工具脚本（评估/对比/LLM pipeline）
│   │   ├── compare_offline_checkpoints.py
│   │   ├── compare_models_with_hprs.py
│   │   ├── run_online_llm_loop.py
│   │   ├── run_llm_pipeline.py
│   │   ├── llm_propose_patch.py
│   │   ├── llm_apply_patch.py
│   │   └── llm_summarize_run.py
│   ├── logs/                         # 训练日志、模型、LLM 分段产物
│   ├── simulator_vec_env.py          # 与仿真对接的 VecEnv 实现
│   ├── evaluate_agent.py             # 单模型评估
│   └── evaluate_test_cases.py        # 测试用例评估
├── auto-shaping/                     # HPRS 奖励塑形库
│   ├── configs/
│   │   └── warehouse.yaml            # 仓储任务奖励规范（核心）
│   ├── auto_shaping/
│   │   ├── hprs_vec_wrapper.py       # HPRS 在 VecEnv 中的注入点
│   │   ├── spec/reward_spec.py       # 奖励规范解析
│   │   ├── parser/                   # 语法/解析器
│   │   └── ...                       # 其他 shaping 实现（pam/rpr/tltl/bhnr）
│   ├── tests/                        # auto-shaping 单测
│   └── run.py
├── environment/simulator/            # Unity 仿真程序（二进制，大体积）
│   ├── v1.0.0/
│   ├── v1.0.1/
│   └── v1.0.2/
├── external/stable-baselines3/       # vendor 第三方库（SB3）
├── Docker/                           # Dockerfile 与启动脚本
├── resources/                        # 图片资源
└── testcases/
    └── warehouse_test_cases.yaml     # 测试场景配置
```

## 3. 关键目录职责（按开发视角）

### 3.1 `agent/`：主业务层

- `agent/config/`：所有实验入口的参数中心。
- `agent/expert/`：生成专家数据集，给离线 TD3-BC 使用。
- `agent/td3/td3bc/`：离线训练。
- `agent/td3/online_td3/`：在线训练（可从离线 checkpoint warm start）。
- `agent/tools/`：实验编排与分析工具，尤其是 LLM 外环 patch 流程。
- `agent/logs/`：运行产物集中地，包含模型、monitor、对比图、LLM 分段记录。

### 3.2 `auto-shaping/`：奖励塑形层

- `configs/warehouse.yaml`：仓储任务奖励定义的事实来源（source of truth）。
- `auto_shaping/hprs_vec_wrapper.py`：把 HPRS 奖励组合进环境 step 的关键位置。
- `spec/reward_spec.py`：读取并解释 reward spec。

### 3.3 `environment/simulator/`：仿真运行时

- 存放 Unity 构建产物（Linux/Windows，多版本）。
- 训练脚本通过网络接口与其通信，不直接修改其二进制内容。

### 3.4 `external/stable-baselines3/`：第三方依赖（vendor）

- 用于复用 RL 组件或参考实现。
- 常规业务开发尽量在 `agent/` 与 `auto-shaping/` 完成，减少对 vendor 目录改动。

## 4. 训练与评估调用链（你最常用的路径）

### 4.1 离线到在线主链路

1. 专家采集：`python agent/expert/run_expert.py --config agent/config/expert_collection.yaml`
2. 离线训练：`python agent/td3/td3bc/train_td3bc.py --config agent/config/offline_td3_bc.yaml`
3. 在线训练：`python agent/td3/online_td3/train_online_td3.py --mode warm_start --config agent/config/online_td3_baseline.yaml`
4. 模型对比：`python agent/tools/compare_models_with_hprs.py --config agent/config/evaluate_compare_models.yaml --verbose`

### 4.2 LLM 外环链路

入口：`agent/tools/run_online_llm_loop.py`

每个 segment 的流程：
1. 先跑一段在线训练
2. `llm_summarize_run.py` 汇总 monitor 指标
3. `llm_propose_patch.py` 产出受限 patch（JSON）
4. `llm_apply_patch.py` 应用到 HPRS YAML
5. 对比新旧配置评估结果
6. 满足验收规则则接受，否则回滚到上一版

## 5. 日志与产物如何看

`agent/logs/` 下常见内容：
- `models/`：训练得到的 checkpoint
- `runs/`：TensorBoard 训练日志
- `compare_models/`：最终模型对比的 csv 与图
- `llm_runs/seg_xx/`：每一段 LLM 调参证据链（patch、diff、summary、accept/reject）

如果你要做实验复盘，优先看：
1. `agent/logs/llm_runs/llm_loop_summary.json`
2. `agent/logs/compare_models/compare_models.csv`
3. 对应 `seg_xx` 目录下的 `llm_patch.json`、`accepted_hprs.yaml`、`reject_reason.txt`

## 6. 最小上手顺序

1. 先读：`README.md`
2. 再读：`docs/Methodology_CN.md`
3. 如果你在推进论文方向，再读：`docs/ReGoRL_Paper_Outline_CN.md`
4. 然后读：`agent/config/*.yaml`（先从 `offline_td3_bc.yaml` 和 `online_td3_llm.yaml` 开始）
5. 最后按代码顺序看：
   - `agent/expert/run_expert.py`
   - `agent/td3/td3bc/train_td3bc.py`
   - `agent/td3/online_td3/train_online_td3.py`
   - `agent/tools/run_online_llm_loop.py`

## 7. 常见开发边界

- 奖励塑形调参优先改：`auto-shaping/configs/warehouse.yaml`
- 流程编排与验收策略改：`agent/tools/run_online_llm_loop.py`
- 算法核心改：`agent/td3/` 下对应实现
- 仿真资源（`environment/simulator/`）通常不在训练实验里直接修改
- 第三方库（`external/stable-baselines3/`）除非必要尽量不动
