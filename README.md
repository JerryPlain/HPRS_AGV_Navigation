# LLM-Assisted Hierarchical Reward Shaping for Offline-to-Online RL in Warehouse Navigation

[![RL](https://img.shields.io/badge/RL-TD3%20%7C%20TD3--BC-0A7EA4)](./VTPRL/agent)
[![Shaping](https://img.shields.io/badge/Reward-HPRS%20(LLM%20optional)-2E8B57)](./VTPRL/auto-shaping/configs/warehouse.yaml)
[![Simulator](https://img.shields.io/badge/Simulator-Unity-orange)](./VTPRL/environment/simulator)
[![Container](https://img.shields.io/badge/Runtime-Docker-2496ED)](./VTPRL/Docker)

End-to-end AGV warehouse navigation training pipeline:
- Expert trajectory collection (`A* + DWA`)
- Offline pretraining (`TD3-BC`)
- Online finetuning (`TD3`)
- Optional LLM-assisted HPRS reward shaping

## Quick Navigation
- [Core Workflow](#core-workflow)
- [Environment Setup](#environment-setup)
- [Run Training Pipeline](#run-training-pipeline)
- [Monitoring](#monitoring)
- [Evaluation](#evaluation)
- [Project Layout](#project-layout)

## Core Workflow

```text
Expert Collection (A* + DWA)
            |
            v
Offline TD3-BC Pretraining
            |
            v
Checkpoint Selection
            |
            v
Online TD3 (scratch / warm_start)
            |
            +--> Optional LLM Loop (HPRS reward patch -> validation -> accept/reject)
```

## Environment Setup

All commands below assume you are inside `VTPRL`:

```bash
cd HPRS_AGV_Navigation/VTPRL
```

### 1) Enable Qt/X11 panel visualization

```bash
export DISPLAY=":1"
xhost + local:
```

### 2) Build Docker image

```bash
docker build . -t vtprl_image -f Docker/Dockerfile_Python310
```

### 3) Start Docker container

```bash
docker run --rm -it \
  --name vtprl_container \
  --gpus all \
  -e DISPLAY \
  -v $(pwd):/home/vtprl:rw \
  -v $(pwd)/external/stable-baselines3:/home/repos/stable-baselines3:ro \
  --privileged \
  --net="host" \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  vtprl_image:latest \
  /home/startup.sh
```

### 4) Configure simulator IP

Set `ip_address` in `agent/config`:
- Linux: `localhost`
- Windows: `host.docker.internal`

### 5) Launch Unity simulator (outside container)

```bash
cd HPRS_AGV_Navigation/VTPRL/environment/simulator/v1.0.2/Linux
./VTPRL-Simulator.x86_64
```

## Run Training Pipeline

### 1) Collect expert trajectories (A* + DWA)

Set `target_episodes` in `agent/config/expert.yaml`, then run:

```bash
python agent/expert/run_expert.py --config agent/config/expert_collection.yaml
```

### 2) Train offline TD3-BC

```bash
python agent/td3/td3bc/train_td3bc.py --config agent/config/offline_td3_bc.yaml
```

### 3) Evaluate offline checkpoints and pick best warm-start model

```bash
python -m pip install matplotlib
python agent/tools/compare_offline_checkpoints.py \
  --config agent/config/evaluate_agent.yaml \
  --checkpoints \
    ./agent/logs/models/td3_bc_offline_step_10000 \
    ./agent/logs/models/td3_bc_offline_step_20000 \
    ./agent/logs/models/td3_bc_offline_step_30000 \
    ./agent/logs/models/td3_bc_offline_step_40000 \
  --verbose
```

### 4) Install optional LLM dependencies

```bash
python -m pip install -r /home/vtprl/requirements_llm.txt
```

### 5) Train online TD3 baseline

From scratch:

```bash
python agent/td3/online_td3/train_online_td3.py \
  --mode scratch \
  --config agent/config/online_td3_baseline.yaml
```

Warm-start from offline model:

```bash
python agent/td3/online_td3/train_online_td3.py \
  --mode warm_start \
  --config agent/config/online_td3_baseline.yaml
```

### 6) Train online TD3 with LLM-assisted HPRS

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

### 7) One-click baseline vs LLM-HPRS comparison

```bash
python agent/tools/run_online_baseline_and_llm.py \
  --mode warm_start \
  --segments 10 \
  --segment_steps 10000 \
  --llm_model Qwen/Qwen2.5-3B-Instruct \
  --val_episodes 10 \
  --accept_delta 0 \
  --accept_reward_delta 0.1 \
  --accept_collision_delta 0 \
  --val_verbose
```

## Monitoring

Offline TD3-BC:

```bash
tensorboard --logdir ./agent/logs/runs/TD3_BC_Offline
```

Online TD3 baseline:

```bash
tensorboard --logdir ./agent/logs/runs/TD3_Online_Baseline
```

LLM-assisted runs:

```bash
tensorboard --logdir ./agent/logs/llm_runs
```

## Evaluation

Build a more complex evaluation environment, then compare:
- Success rate
- Collision rate
- Mean reward on successful episodes

```bash
python agent/tools/compare_models_with_hprs.py \
  --config agent/config/evaluate_compare_models.yaml \
  --verbose
```

## Project Layout

```text
HPRS_AGV_Navigation/
└── VTPRL/
    ├── agent/                  # expert, offline, online, tools, logs
    ├── auto-shaping/           # HPRS reward specs and wrappers
    ├── environment/simulator/  # Unity simulator binaries
    ├── Docker/                 # Dockerfiles and startup scripts
    ├── docs/                   # methodology and repo guide
    └── testcases/              # evaluation cases
```
