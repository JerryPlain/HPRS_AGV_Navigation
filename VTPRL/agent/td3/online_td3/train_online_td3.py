import os
import sys
import numpy as np
import pickle
import argparse
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from envs.get_env import get_env
from config import Config

from expert.expert_dataset import ExpertDataset
import expert
sys.modules["expert"] = expert
sys.modules["expert.expert_dataset"] = sys.modules["expert.expert_dataset"]

from td3.td3_agent import TD3Agent
from td3.td3bc.td3_bc_agent import TD3BCAgent
from td3.utils.replay_buffer import ReplayBuffer

# --- CHANGE 1: Import the new wrapper ---
from td3.online_td3.egocentric_normalization_wrapper import EgocentricNormalizationWrapper

def evaluate(agent, env, num_episodes=10):
    """
    Evaluates the agent using vectorized environment.
    """
    agent.actor.eval()
    
    episode_rewards = []
    episode_successes = []
    
    obs = env.reset()
    
    current_rewards = np.zeros(env.num_envs)
    
    while len(episode_rewards) < num_episodes:
        action = agent.select_action(obs, noise=0.0)

        next_obs, rewards, dones, infos = env.step(action)
        
        current_rewards += rewards
        
        for i in range(env.num_envs):
            if dones[i]:
                # Episode finished
                episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0.0
                
                # Check success
                is_success = infos[i].get('success', False) or infos[i].get('is_success', False)
                episode_successes.append(1.0 if is_success else 0.0)
        
        obs = next_obs
        
    agent.actor.train()
    
    mean_reward = np.mean(episode_rewards[:num_episodes])
    success_rate = np.mean(episode_successes[:num_episodes])
    
    return mean_reward, success_rate

def load_offline_weights(agent, model_path_prefix):
    actor_path = f"{model_path_prefix}_actor.pth"
    critic_path = f"{model_path_prefix}_critic.pth"

    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"[ONLINE-TD3] Could not find actor weights at {actor_path}")
    
    print(f"[ONLINE-TD3] Loading warm-start weights from:")
    print(f"[ONLINE-TD3]     Actor:  {actor_path}")
    print(f"[ONLINE-TD3]     Critic: {critic_path}")

    agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic.load_state_dict(torch.load(critic_path, map_location=agent.device))
    agent.critic_target.load_state_dict(agent.critic.state_dict())

def load_online_config(path: str) -> dict:
    """Load online training configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="agent/config/td3_online.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="scratch", choices=["scratch", "warm_start"], help="Training mode")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Override pretrained path for warm start")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override dataset path")
    args = parser.parse_args()
    
    # Load configuration from YAML
    config = load_online_config(args.config)
    if args.pretrained_path:
        config["pretrained_path"] = args.pretrained_path
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path
    
    # Extract hyperparameters
    MAX_TIMESTEPS = config['max_timesteps']
    BATCH_SIZE = config['batch_size']
    EVAL_FREQ = config['eval_freq']
    START_STEPS = config['start_steps']

    # Setup Logging
    run_name = f"TD3_{args.mode}_Online_EGO" # Added _EGO to run name
    writer = SummaryWriter(f"{config['log_dir']}/{run_name}")
    if not os.path.exists(config['models_dir']): 
        os.makedirs(config['models_dir'])

    # ---------------------------------- CONFIGURATION ----------------------------------
    main_config = Config()

    agent_cfg = main_config.agent_dict
    root = main_config.root_dict
    sim = main_config.simulation_dict
    gym = main_config.gym_environment_dict
    manip_env = main_config.manipulator_environment_dict
    warehouse_env = main_config.warehouse_environment_dict
    observation = main_config.observation_dict
    reward_dict = gym['manipulator_gym_environment']['reward']

    dataset_path = config['dataset_path']
    hprs_config = config['hprs_config']

    # Ensure online runs write monitor.csv under the run-specific log_dir
    agent_cfg["log_dir"] = config["log_dir"]

    print("=" * 60)
    print(f"[TD3 Online] Configuration summary")
    print(f"  Config file      : {args.config}")
    print(f"  Mode             : {args.mode}")
    print(f"  Observation      : EGOCENTRIC (Polar)")
    print(f"  Max timesteps    : {MAX_TIMESTEPS}")
    print(f"  Dataset path     : {config['dataset_path']}")
    print("=" * 60)

    # Load Statistics for Normalization
    print(f"[ONLINE-TD3] Loading dataset stats from {dataset_path}...")
    dataset = ExpertDataset(dataset_path)
    
    stats = dataset.stats
    
    action_max = 1.0

    # ---------------------------------- ENV SETUP ----------------------------------
    # Common Wrapper Args
    wrapper_kwargs = {
        "stats": stats,
        "normalize_obs": True,          # Calculates Polar coords + manual scaling
        "unnormalize_actions": True,    # Agent [-1,1] -> Env [Physical]
    }

    # Train Env (Rewards Normalized only if warm starting or desired)
    train_wrapper_kwargs = wrapper_kwargs.copy()
    #train_wrapper_kwargs["normalize_rewards"] = (args.mode == "warm_start")
    
    # --- CHANGE 3: Use EgocentricNormalizationWrapper ---
    train_env = get_env(
        agent_cfg, root, sim, gym, manip_env, warehouse_env, 
        observation, reward_dict, agent_cfg["log_dir"],
        hprs_config,
        additional_wrapper=EgocentricNormalizationWrapper, # <--- SWITCHED
        wrapper_kwargs=train_wrapper_kwargs
    )

    state_dim = train_env.observation_space.shape[0] 
    action_dim = train_env.action_space.shape[0]

    print(f"[ONLINE-TD3] State Dim: {state_dim} (Should be 5) | Action Dim: {action_dim}")

    print("[ONLINE-TD3] Environments initialized.")

    # ---------------------------------- AGENT SETUP ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3BCAgent(
        state_dim,
        action_dim,
        action_max,
        device,
        lr=5e-5,
        alpha=2.5,
        policy_delay=4,
    )
    buffer = ReplayBuffer(state_dim, action_dim, max_size=1_000_000, device=device)

    # --- WARM START: Load Weights & Pre-fill Buffer ---
    if args.mode == "warm_start":
        load_offline_weights(agent, config['pretrained_path'])
        
        print(f"[ONLINE-TD3] Pre-filling replay buffer with offline data ({len(dataset.states)} samples)...")
        # Since we called dataset.normalize(mode="ego"), data["states"] is ALREADY 5-dim
        for i in range(len(dataset.states)):
            buffer.add(
                dataset.states[i], 
                dataset.actions[i], 
                dataset.rewards[i], 
                dataset.next_states[i], 
                dataset.dones[i]
            )
        print("[ONLINE-TD3] Buffer filled.")

    # ---------------------------------- TRAINING LOOP ----------------------------------
    print(f"Starting Training | Mode: {args.mode} | Device: {device}")

    # Initial Reset
    obs = train_env.reset()
    
    total_timesteps = 0
    best_success = -1.0
    save_every = int(config.get("save_every", 10000))

    # TRACKING VARIABLES
    running_reward = 0.0
    running_success = 0.0
    episodes_count = 0
    
    while total_timesteps < MAX_TIMESTEPS:
        
        # 1. Select Action
        if args.mode == "scratch" and total_timesteps < START_STEPS:
            action = np.random.uniform(-1.0, 1.0, size=(train_env.num_envs, action_dim))
        else:
            noise = 0.1 if args.mode == "warm_start" else 0.2
            action = agent.select_action(obs, noise=noise)

        # 2. Step Environment
        next_obs, rewards, dones, infos = train_env.step(action)
        
        for i in range(train_env.num_envs):
            
            if dones[i]:
                # Episode finished
                
                # NOTE: EgocentricWrapper handles the observation reset internally,
                # so obs[i] (before reset) is the closest we have to terminal state.
                real_next_obs = obs[i].copy()

                is_success = infos[i].get('success', False) or infos[i].get('is_success', False)
                
                running_success += 1.0 if is_success else 0.0
                episodes_count += 1
                
                if episodes_count % 1 == 0:
                    avg_succ = running_success / 1.0
                    print(f"Step {total_timesteps} | Recent Success Rate: {avg_succ:.2f}")
                    writer.add_scalar("Train/Success_Rate_with_Noise", avg_succ, total_timesteps)
                    
                    if avg_succ >= best_success:
                        best_success = avg_succ
                        agent.save(f"{config['models_dir']}/{run_name}_best")
                        print("--> New Best Model (Training)")
                    
                    running_success = 0.0
                    episodes_count = 0
            else:
                real_next_obs = next_obs[i]

            buffer.add(obs[i], action[i], rewards[i], real_next_obs, dones[i])
        
        obs = next_obs
        total_timesteps += train_env.num_envs

        # 4. Train Agent
        start_update_step = START_STEPS if args.mode == "scratch" else 1000
        
        if total_timesteps >= start_update_step:
            
            c_loss, a_loss, _ = agent.train(buffer, BATCH_SIZE)
            
            if total_timesteps % 100 == 0:
                writer.add_scalar("Train/Critic_Loss", c_loss, total_timesteps)
                if a_loss is not None:
                    writer.add_scalar("Train/Actor_Loss", a_loss, total_timesteps)
                    writer.add_scalar("Train/Total_Loss", c_loss + a_loss, total_timesteps)

        if save_every > 0 and total_timesteps % save_every == 0:
            agent.save(f"{config['models_dir']}/{run_name}_step_{total_timesteps}")
            print(f"[TD3 Online] Saved checkpoint: {config['models_dir']}/{run_name}_step_{total_timesteps}")

    # Save final checkpoint for chaining runs
    agent.save(f"{config['models_dir']}/{run_name}_last")
    print(f"[TD3 Online] Saved last checkpoint: {config['models_dir']}/{run_name}_last")

    train_env.close()
    writer.close()

if __name__ == "__main__":
    main()
