# LLM-Assisted Hierarchical Reward Shaping for Offline-to-Online RL in AGV Warehouse Navigation

This project provides an end-to-end training pipeline for AGV warehouse navigation:
- Expert trajectory collection (A* + DWA)
- Offline RL pretraining (TD3-BC)
- Online RL finetuning (TD3)
- Optional LLM-assisted hierarchical potential-based reward shaping (HPRS)

## Table of Contents
- [Environment Setup](#environment-setup)
- [Run Pipeline](#run-pipeline)
- [Monitoring](#monitoring)
- [Evaluation](#evaluation)

## Environment Setup

### 1) Enable Qt/X11 panel visualization
```bash
export DISPLAY=":1"
xhost + local:
```

### 2) Build Docker image
Run inside the `VTPRL` directory:
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

### 4) Set simulator IP address
Update `ip_address` in `agent/config` for Unity simulation:
- Linux: `localhost`
- Windows: `host.docker.internal`

### 5) Launch Unity simulator
Open a new terminal outside the container:
```bash
cd g3_repo/VTPRL/environment/simulator/v1.0.2/Linux
./VTPRL-Simulator.x86_64
```

## Run Pipeline

### 1) Collect expert trajectories (A* + DWA)
Set `target_episodes` in `agent/config/expert.yaml`, then run:
```bash
python agent/expert/run_expert.py --config agent/config/expert_collection.yaml
```

### 2) Train offline TD3-BC agent
```bash
python agent/td3/td3bc/train_td3bc.py --config agent/config/offline_td3_bc.yaml
```

### 3) Evaluate offline checkpoints
Install plotting dependency:
```bash
python -m pip install matplotlib
```

Compare checkpoints and select the best model for online training:
```bash
python agent/tools/compare_offline_checkpoints.py \
  --config agent/config/evaluate_agent.yaml \
  --checkpoints \
    ./agent/logs/models/td3_bc_offline_step_10000 \
    ./agent/logs/models/td3_bc_offline_step_20000 \
    ./agent/logs/models/td3_bc_offline_step_30000 \
    ./agent/logs/models/td3_bc_offline_step_40000 \
  --verbose
```

### 4) Prepare LLM dependencies (optional)
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

Warm-start from offline data:
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

### 7) Run one-click baseline vs LLM-HPRS comparison
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
Create a more complex evaluation environment, then compare:
- Success rate
- Collision rate
- Mean reward on successful episodes

```bash
python agent/tools/compare_models_with_hprs.py \
  --config agent/config/evaluate_compare_models.yaml \
  --verbose
```
