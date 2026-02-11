import sys
import os
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from agent.td3.online_td3.egocentric_normalization_wrapper import EgocentricNormalizationWrapper
from config import Config
from envs.get_env import get_env


import expert
import expert.expert_dataset
sys.modules["expert"] = expert
sys.modules["expert.expert_dataset"] = expert.expert_dataset

from expert.expert_dataset import ExpertDataset
from td3.td3bc.td3_bc_agent import TD3BCAgent


def evaluate_agent(
    env,
    agent,
    num_episodes=10,
    max_steps=2000,
    verbose=True,
    randomize_seed=False,
    base_seed=0,
):
    """
    Evaluate a trained agent.
    NOTE: Normalization is handled by the Environment Wrapper, not here.
    """
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        if randomize_seed:
            env.seed(base_seed + ep)
        # Wrapper returns Normalized Obs
        obs = env.reset()
        
        # Handle batched observations (n_envs, obs_dim) -> (obs_dim,)
        if len(obs.shape) > 1:
            obs = obs[0]
        
        done = False
        total_reward = 0.0
        step = 0
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Episode {ep + 1}/{num_episodes}")
            print(f"{'='*50}")
        
        while not done and step < max_steps:

            state = obs.astype(np.float32)

            action = agent.select_action(state, noise=0.0)
            
            next_obs, reward, dones, info = env.step([action])
            
            obs = next_obs[0]
            reward_scalar = reward[0]
            done = dones[0]
            info_dict = info[0]
            
            total_reward += reward_scalar
            step += 1
        
        # Episode finished
        success = info_dict.get('success', False)
        # Fallback check if info dict structure differs
        if not success:
             success = info_dict.get('is_success', False)

        episode_rewards.append(total_reward)
        episode_successes.append(1.0 if success else 0.0)
        episode_lengths.append(step)
        
        if verbose:
            print(f"Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"Steps: {step}, Reward: {total_reward:.2f}")
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(episode_successes),
        'num_episodes': num_episodes,
    }
    
    return results


def load_eval_config(path: str) -> dict:
    """Load evaluation configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # ===========================
    # Configuration
    # ===========================

    parser = argparse.ArgumentParser(description="Evaluate a trained TD3-BC agent")
    parser.add_argument(
        "--config",
        type=str,
        default="agent/config/evaluate.yaml",
        help="Path to evaluation config file"
    )
    args = parser.parse_args()
    eval_config = load_eval_config(args.config)

    DATASET_PATH = eval_config['dataset_path']
    MODEL_PATH = eval_config['model_path']
    NUM_EVAL_EPISODES = eval_config['num_episodes']
    HPRS_CONFIG = eval_config['hprs_config']
    RANDOMIZE_SEED = bool(eval_config.get("randomize_seed", False))
    BASE_SEED = int(eval_config.get("base_seed", 0))

    print("=" * 60)
    print("[Evaluation] Configuration Summary")
    print("=" * 60)
    print(f"Config file:    {args.config}")
    print(f"Dataset:        {DATASET_PATH}")
    print(f"Model:          {MODEL_PATH}")
    print(f"Episodes:       {NUM_EVAL_EPISODES}")
    print(f"HPRS config:    {HPRS_CONFIG}")
    print(f"Randomize seed: {RANDOMIZE_SEED} (base={BASE_SEED})")
    print("=" * 60 + "\n")
    
    # ===========================
    # Load Configuration
    # ===========================
    main_config = Config()
    agent_cfg = main_config.agent_dict
    root = main_config.root_dict
    sim = main_config.simulation_dict
    gym = main_config.gym_environment_dict
    manip_env = main_config.manipulator_environment_dict
    warehouse_env = main_config.warehouse_environment_dict
    observation = main_config.observation_dict
    reward_dict = gym['manipulator_gym_environment']['reward']

    gym['num_envs'] = 1
    
    # Create log directory
    Path(agent_cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
    
    # ===========================
    # Load Dataset Statistics
    # ===========================
    print(f"[Eval] Loading dataset statistics from {DATASET_PATH}...")
    try:
        dataset = ExpertDataset(path=DATASET_PATH)
        stats = dataset.stats
        
        state_dim = stats["state_dim"]
        action_dim = stats["action_dim"]
        
        print(f"[Eval] Dataset info:")
        print(f"       Total state dim:  {state_dim}")
        print(f"       Action dim:       {action_dim}")

    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        exit(1)
    
    # ===========================
    # Create Environment
    # ===========================
    print(f"[Eval] Creating environment...")
    

    env = get_env(
            agent_cfg, root, sim, gym, manip_env, warehouse_env,
            observation, reward_dict, agent_cfg["log_dir"],
            HPRS_CONFIG,
            additional_wrapper=EgocentricNormalizationWrapper, 
            wrapper_kwargs={
                "stats": stats,
                "normalize_obs": True,       
                "unnormalize_actions": True,
            }
        )
    print(f"[Eval] Environment observation space: {env.observation_space.shape}")
    
    # ===========================
    # Initialize Agent
    # ===========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")
    
    agent = TD3BCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,  
        device=device
    )
    
    # ===========================
    # Load Model Weights
    # ===========================
    print(f"[Eval] Loading model from {MODEL_PATH}...")
    try:
        agent.load(MODEL_PATH)
        print("[Eval] Model loaded successfully.")
    except Exception as e:
        print(f"[Error] Could not load model: {e}")
        exit(1)
    
    # ===========================
    # Run Evaluation
    # ===========================
    print(f"\n{'='*60}")
    print(f"Starting Evaluation: {NUM_EVAL_EPISODES} episodes")
    print(f"{'='*60}")
    

    results = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=NUM_EVAL_EPISODES,
        verbose=True,
        randomize_seed=RANDOMIZE_SEED,
        base_seed=BASE_SEED,
    )
    
    # ===========================
    # Print Results
    # ===========================
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Episodes:      {results['num_episodes']}")
    print(f"Success Rate:  {results['success_rate']:.1%} ({int(results['success_rate'] * results['num_episodes'])}/{results['num_episodes']})")
    print(f"Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length:   {results['mean_length']:.1f} steps")
    print(f"{'='*60}\n")
    
    env.close()
