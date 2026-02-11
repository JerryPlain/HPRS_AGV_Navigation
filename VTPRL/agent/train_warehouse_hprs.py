import os
import numpy as np
from pathlib import Path

from config import Config
from simulator_vec_env import SimulatorVecEnv

# envs
from envs.warehouse_unity_env import WarehouseUnityEnv

# utils
from utils.helpers import set_seeds

# HPRSVecWrapper
from auto_shaping.hprs_vec_wrapper import HPRSVecWrapper

# reward spec from Auto-Shaping
from auto_shaping.spec.reward_spec import RewardSpec
from auto_shaping.utils.hprs_tb_callback import HPRSTensorboardCallback

# expert dataset (optional replay buffer preload)
from expert.expert_dataset import ExpertDataset

# SB3
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import PPO, TD3

"""
Build and return a fully-wrapped vectorized environment instance based on the provided configuration.
Wrapper stack:
Base Gymnasium environment (warehouse_unity_env)-> SimulatorVecEnv -> HPRSVecWrapper (if training/evaluation) -> VecMonitor (logging)
"""
def get_env(agent, root, sim, gym_cfg, manip_env, warehouse_env, observation, reward_dict, log_dir):

    env_key = gym_cfg['env_key']

    # -----------------------------------------------
    # Create single env function, id used for seeding
    # -----------------------------------------------

    def create_env(id=0):

        # -----------------------------------------------
        # Warehouse environment
        # -----------------------------------------------
        # Here is single env creation for warehouse_unity_env, HPRSVecWrapper applied later

        if env_key == 'warehouse_unity_env':
            env = WarehouseUnityEnv(
                max_time_steps=gym_cfg['max_time_step'],
                env_id=id,
                gym_config=gym_cfg['warehouse_gym_environment'],
                warehouse_config=warehouse_env,
                observation_config=observation,
            )

        env.seed((id * 150) + (id + 11)) # Unique seed per env instance for reproducibility e.g., env 0 -> seed 11, env 1 -> seed 161, env 2 -> seed 311, etc.
        return env

    num_envs = gym_cfg['num_envs'] # Number of parallel environments

    # Create a list of environment-creation functions for the vectorized wrapper
    env = [create_env for i in range(num_envs)]

    # Instantiate the vectorized environment wrapper
    env = SimulatorVecEnv(env, agent, root, sim, gym_cfg, manip_env, reward_dict=reward_dict)

    # -----------------------------------------------
    # Apply HPRSVecWrapper if in training or evaluation mode
    # -----------------------------------------------
    print("simulation_mode:", agent.get('simulation_mode'))
    if agent.get('simulation_mode') in ['train', 'evaluate']: # Apply HPRS only in train/evaluate modes
        spec_file = "auto-shaping/configs/warehouse.yaml" # HPRS reward specification file for warehouse env
        reward_spec = RewardSpec.from_yaml(spec_file) # Load reward specification
        print("[HPRS] Loaded spec:", spec_file) 

        # Use PPO gamma as default fallback for other algorithms
        hprs_gamma = agent.get("td3_gamma", agent.get("ppo_gamma", 0.99))
        env = HPRSVecWrapper(env, reward_spec, gamma=hprs_gamma, debug_print=True)

    ##############################
    # Apply VecMonitor for logging
    ##############################
    env = VecMonitor(
        env,
        log_dir,
        info_keywords=(
            # episode outcome
            'hprs/success',
            'hprs/collision',

            # potentials
            "hprs/phi_s", "hprs/phi_t", "hprs/phi_c",

            # PBRS Deltas (actually shape rewards)
            "hprs/delta_phi_s", "hprs/delta_phi_t", "hprs/delta_phi_c",
            "hprs/shaping_total",

            # rewards
            "hprs/base_reward",
            "hprs/shaped_reward_debug",

            # state diagnostics
            "hprs/dist_to_goal",
            "hprs/delta_yaw",
            "hprs/min_laser",
            "hprs/robot_v",
            "hprs/robot_omega")
        )

    return env

def _preload_td3_replay_buffer(model, dataset_path: str, normalize: bool) -> None:
    if not dataset_path:
        return
    if not os.path.exists(dataset_path):
        print(f"[TD3] Dataset not found: {dataset_path}")
        return

    print(f"[TD3] Preloading replay buffer from: {dataset_path}")
    dataset = ExpertDataset.load(dataset_path)

    if normalize:
        data = dataset.normalize()
        states = data["states"]
        next_states = data["next_states"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
    else:
        states = np.array([d["obs"] for d in dataset], dtype=np.float32)
        next_states = np.array([d["next_obs"] for d in dataset], dtype=np.float32)
        actions = np.array([d["action"] for d in dataset], dtype=np.float32)
        rewards = np.array([d["reward"] for d in dataset], dtype=np.float32)
        dones = np.array([d["done"] for d in dataset], dtype=np.float32)

    n_envs = model.n_envs
    total = states.shape[0]
    usable = total - (total % n_envs)
    if usable == 0:
        print("[TD3] Dataset too small for current num_envs, skipping preload.")
        return

    for i in range(0, usable, n_envs):
        obs = states[i:i + n_envs]
        next_obs = next_states[i:i + n_envs]
        action = actions[i:i + n_envs]
        reward = rewards[i:i + n_envs]
        done = dones[i:i + n_envs]
        infos = [{} for _ in range(n_envs)]
        model.replay_buffer.add(obs, next_obs, action, reward, done, infos)

    print(f"[TD3] Preloaded {usable} transitions into replay buffer.")

def _offline_td3_updates(model, gradient_steps: int, batch_size: int, tb_log_name: str) -> None:
    if gradient_steps <= 0:
        return
    if model.replay_buffer is None or model.replay_buffer.size() == 0:
        print("[TD3] Replay buffer empty, skipping offline updates.")
        return

    # Ensure logger is initialized before calling train()
    if model.logger is None:
        model._setup_learn(
            total_timesteps=0,
            eval_env=None,
            callback=None,
            reset_num_timesteps=True,
            tb_log_name=tb_log_name
        )
    model._current_progress_remaining = 1.0

    print(f"[TD3] Running offline updates: gradient_steps={gradient_steps}, batch_size={batch_size}")
    model.train(gradient_steps=gradient_steps, batch_size=batch_size)

if __name__ == "__main__":

    # ---------------------------------- CONFIGURATION ----------------------------------

    main_config = Config()

    # Structured config handles
    agent = main_config.agent_dict
    root = main_config.root_dict
    sim = main_config.simulation_dict
    gym = main_config.gym_environment_dict
    manip_env = main_config.manipulator_environment_dict
    warehouse_env = main_config.warehouse_environment_dict
    observation = main_config.observation_dict

    # Reward dict (terminal reward used in SimulatorVecEnv)
    reward_dict = gym['manipulator_gym_environment']['reward']

    # Create new folder if not exists for logging #
    Path(agent["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Build env
    env = get_env(agent, root, sim, gym, manip_env, warehouse_env, observation, reward_dict, agent["log_dir"])

    # ---------------------------------- TRAINING MODE ----------------------------------

    if agent['simulation_mode'] == 'train':

        # Set global seeds and PPO seed
        ppo_seed = set_seeds(sim["random_seed"])  # 'random_seed' is 256

        algo = agent.get("model", "PPO").upper()

        if algo == "TD3":
            # TD3 needs action noise for exploration in continuous control
            n_actions = env.action_space.shape[0]
            noise_sigma = agent.get("td3_action_noise_sigma", 0.1)
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_sigma * np.ones(n_actions)
            )
            pretrained_path = agent.get("td3_pretrained_path", "")
            if pretrained_path:
                if not os.path.exists(pretrained_path):
                    raise FileNotFoundError(f"[TD3] Pretrained checkpoint not found: {pretrained_path}")
                model = TD3.load(
                    pretrained_path,
                    env=env,
                    seed=ppo_seed,
                    tensorboard_log=agent["log_dir"],
                    action_noise=action_noise,
                    verbose=1
                )
            else:
                model = TD3(
                    policy="MlpPolicy",
                    env=env,
                    seed=ppo_seed,
                    tensorboard_log=agent["log_dir"],
                    gamma=agent.get("td3_gamma", agent.get("ppo_gamma", 0.99)),
                    action_noise=action_noise,
                    verbose=1
                )
            _preload_td3_replay_buffer(
                model,
                dataset_path=agent.get("td3_dataset_path", ""),
                normalize=agent.get("td3_dataset_normalize", False)
            )
            _offline_td3_updates(
                model,
                gradient_steps=agent.get("td3_offline_gradient_steps", 0),
                batch_size=agent.get("td3_offline_batch_size", 256),
                tb_log_name=agent["tb_log_name"]
            )
        else:
            # Default to PPO
            model = PPO(
                policy="MlpPolicy",
                env=env,
                seed=ppo_seed,
                tensorboard_log=agent["log_dir"],
                gamma=agent.get("ppo_gamma", 0.99),  # Discount factor for PPO
                verbose=1
            )

        print("===================================================")
        print("Starting training for " + str(agent["total_timesteps"]) + " timesteps")
        print("===================================================")

        model.learn(
            total_timesteps=agent["total_timesteps"],
            reset_num_timesteps=True,
            tb_log_name=agent["tb_log_name"],
            log_interval=2,
            callback=HPRSTensorboardCallback(log_every=50) # Log HPRS signals to TensorBoard every 50 timesteps
        )
        print("===================================================")
        print("Training ended. Saving checkpoint at: " + agent["log_dir"])
        print("===================================================")

        # Save model
        save_name = "td3_trained" if algo == "TD3" else "ppo_trained"
        model.save(os.path.join(agent["log_dir"], save_name))

        del model  # Delete model instance

    # ---------------------------------- EVALUATION MODE (RL policy) ----------------------------------

    elif agent['simulation_mode'] == 'evaluate':
        print("===================================================")
        print("RL-based evaluation")
        print("===================================================")

        algo = agent.get("model", "PPO").upper()
        if algo == "TD3":
            model = TD3.load(os.path.join(agent["log_dir"], "td3_trained"))
        else:
            model = PPO.load(os.path.join(agent["log_dir"], "ppo_trained"))
        model.policy.set_training_mode(False)

        obs = env.reset()
        for x in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)

            # Render (if enabled and not joint-vel env)
            if gym['manipulator_gym_environment']['dart']['enable_dart_viewer'] and gym['env_key'] != 'iiwa_joint_vel':
                env.render()


    # ---------------------------------- MODEL-BASED (Warehouse AGV) ----------------------------------

    elif agent['simulation_mode'] == 'evaluate_model_based' and gym['env_key'] == 'warehouse_unity_env':
        print("===================================================")
        print("Warehouse model-based evaluation")
        print("===================================================")

        obs = env.reset()
        episode_rewards = []
        for _ in range(3):
            cum_reward = 0.0
            while True:
                # Generate action using P-controller
                action = np.reshape(
                    env.env_method('action_by_p_control', 1.0, 2.0),
                    (gym['num_envs'], env.action_space.shape[0])
                )

                obs, rewards, dones, info = env.step(action)
                cum_reward += rewards

                if dones.any():
                    episode_rewards.append(cum_reward)
                    break

        mean_reward = np.mean(episode_rewards)
        print("===================================================")
        print("Mean reward: " + str(mean_reward))
        print("===================================================")

    # ---------------------------------- INVALID MODE ----------------------------------
    else:
        print("Invalid simulation_mode or configuration detected - aborting.")
