# LLM-Assisted Hierarchical Reward Shaping for Offline-to-Online RL in AGV Warehouse Navigation

## Add authorization for Qt/X11 for panel visualization
```bash
export DISPLAY=":1"
xhost + local:
```

## Build and Start Docker Container
Inside `VTPRL` folder execute `docker build . -t vtprl_image -f Docker/Dockerfile_Python310`
To start the container run:
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

## IP_Address Set Up
change ip_address in agent/config for unity simulation 
```bash
- for Linux: 'localhost'
- for Windows: 'host.docker.internal'
```

## Launch Unity Simulator
Note: Open a new terminal (outside the container) to run the simulation in Unity.
```bash
cd g3_repo/VTPRL/environment/simulator/v1.0.2/Linux
./VTPRL-Simulator.x86_64
```

## Run Pipeline
### Expert trajectory collection using A* and DWA (Dynamic Window Approach)
choose the target_episodes in agent/config/expert.yaml and run the following command:
```bash
python agent/expert/run_expert.py --config agent/config/expert_collection.yaml
```

### Train offline TD3-BC agent on expert data
```bash
python agent/td3/td3bc/train_td3bc.py --config agent/config/offline_td3_bc.yaml
```

#### Tensorboard visualization of offline RL
```bash
tensorboard --logdir ./agent/logs/runs/TD3_BC_Offline
```

#### Evaluate the offline RL
compare the checkpoints and choose the best one for online TD3 RL
```bash
python -m pip install matplotlib
```

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

### Train TD3 online RL Agent with Hierarchical potential-based Reward shaping with LLM Assistance
```bash
python -m pip install -r /home/vtprl/requirements_llm.txt
```

#### Run online RL TD3 with HPRS from scratch
```bash
python agent/td3/online_td3/train_online_td3.py \
  --mode scratch \
  --config agent/config/online_td3_baseline.yaml
```

#### Run online RL TD3 with HPRS with warm-started offline dataset
```bash
python agent/td3/online_td3/train_online_td3.py \
  --mode warm_start \
  --config agent/config/online_td3_baseline.yaml
```

#### Run LLM-Assisted online RL TD3 with warm-started offline dataset
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

#### Run the Online RL with and without LLM-Assisted HPRS once
Use the following command to get and compare the results
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

#### Tensorboard visualization
```bash
tensorboard --logdir ./agent/logs/runs/TD3_Online_Baseline
tensorboard --logdir ./agent/logs/llm_runs
```

### Evaluate and Comparison
Set up a more complex new environment for evaluation between with or without LLM HPRS
Compare the success rate, collision rate, and success mean reward
```bash
python agent/tools/compare_models_with_hprs.py \
  --config agent/config/evaluate_compare_models.yaml \
  --verbose
```