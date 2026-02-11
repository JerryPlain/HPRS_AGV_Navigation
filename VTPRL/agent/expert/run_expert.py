"""
Pipeline:
1. Initialize vectorized warehouse navigation environment with multiple parallel environments.
2. For each environment, plan an initial path using A* algorithm and DWA to get expert actions.
3. In a loop, for each environment:
   - Generate expert actions using DWA.
   - Add exploration noise to actions.
   - Step the environment with noisy actions.
   - Store transitions (obs, action, reward, done, next_obs).
   - On episode completion, check for success and add successful episodes to the dataset.
"""
import sys
import os
import yaml
import pickle
import argparse
import numpy as np
from pathlib import Path

# Add parent directory (agent/) to Python path so we can import config, simulator_vec_env, etc.
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import Config

from envs.get_env import get_env # get vectorized environment

def load_expert_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    
    # ===========================
    # Parse Command Line Arguments
    # ===========================
    parser = argparse.ArgumentParser(description="Expert data collection for warehouse navigation")
    parser.add_argument(
        "--config",
        type=str,
        default="agent/config/expert_collection.yaml",
        help="Path to expert collection config file"
    )
    args = parser.parse_args()

    # ===========================
    # Load Configurations
    # ===========================
    expert_cfg = load_expert_config(args.config)
    hprs_config = expert_cfg['hprs_config'] # use warehouse.yaml

    main_config = Config()
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
    frames_dir = Path(agent["log_dir"]) / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Instantiate vectorized environment
    env = get_env(agent, root, sim, gym, manip_env, warehouse_env, observation, reward_dict, agent["log_dir"], hprs_config)

    # Initialize expert dataset
    dataset = []
    episode_rewards = []
    episode_successes = 0
    target_episodes = expert_cfg['target_episodes']
    print_interval = int(expert_cfg.get("print_interval", 100))

    num_envs = gym['num_envs']

    # Per-environment tracking
    episode_data = [[] for _ in range(num_envs)]  # Stores transitions for each env
    episode_count = 0  # Total completed episodes
    episode_successes = 0  # Total successful episodes
    env_episode_idx = [0 for _ in range(num_envs)]  # Episode index per env
    env_step_count = [0 for _ in range(num_envs)]  # Steps per env

    print(f"EXPERT: Parallel envs: {num_envs}")
    print(f"EXPERT: Target episodes: {target_episodes}")
    
    obs = env.reset()
    total_steps = 0

    # for each environment, plan initial path using A*
    # waypoints will be used by DWA for expert actions
    for env_idx in range(num_envs):
        a_star_path = env.env_method('plan_global_path_astar', indices=env_idx)[0]
        
        # Skip if planning failed
        if a_star_path is None or len(a_star_path) == 0:
            print(f"EXPERT: [Env {env_idx}] Episode {env_episode_idx[env_idx] + 1}: Path planning failed, continuing...")
            # Don't plan, just let it continue with P-controller
            # The episode will likely fail but won't be added to dataset
        else:
            print(f"EXPERT: [Env {env_idx}] Episode {env_episode_idx[env_idx] + 1}: "
                f"Planned path with {len(a_star_path)} waypoints")  

    # Main vectorized collection loop
    while episode_count < target_episodes:
        # Generate expert actions for all environments by using DWA
        expert_actions = np.reshape(
            env.env_method('action_by_p_control', 1.0, 2.0),
            (num_envs, env.action_space.shape[0])
        )
        
        # Add exploration noise, clip to action space (set to 0.0 for clean expert)
        noise = np.random.normal(0, 0.0, size=expert_actions.shape)
        actions = np.clip(
            expert_actions + noise, # adding noise for exploration
            env.action_space.low,
            env.action_space.high
        )
        
        # Step all environments
        next_obs, rewards, dones, infos = env.step(actions)
        total_steps += num_envs

        if print_interval > 0 and (total_steps % print_interval) == 0:
            # Print a lightweight progress line using env 0 info when available
            info0 = infos[0] if infos else {}
            dist = info0.get("hprs/dist_to_goal", None)
            succ = info0.get("hprs/success", info0.get("success", None))
            coll = info0.get("hprs/collision", info0.get("collision", None))
            extra = []
            if dist is not None:
                extra.append(f"dist_to_goal={dist:.3f}")
            if succ is not None:
                extra.append(f"success={succ}")
            if coll is not None:
                extra.append(f"collision={coll}")
            extra_str = " | " + ", ".join(extra) if extra else ""
            print(f"EXPERT: Steps={total_steps} | Episodes={episode_count}/{target_episodes} | Success={episode_successes}{extra_str}")
        
        # Process each environment, store transitions
        for env_idx in range(num_envs):
            if episode_count >= target_episodes:
                break
        
            if dones[env_idx]:
                # The episode ended. We don't have terminal_observation.
                # We use the LAST valid observation we saw (before the step)
                # This represents the state where the crash/success happened.
                save_next_obs = obs[env_idx]
            else:
                save_next_obs = next_obs[env_idx]

            episode_data[env_idx].append({
                "obs": obs[env_idx].tolist(),
                "action": actions[env_idx].tolist(),
                "reward": rewards[env_idx],
                "done": float(dones[env_idx]),
                "next_obs": save_next_obs.tolist(),
            })
                
            env_step_count[env_idx] += 1
            
            # Check if episode finished
            if dones[env_idx]:
                episode_count += 1
                env_episode_idx[env_idx] += 1
                
                # Check success
                if infos[env_idx].get('success', False):
                    episode_successes += 1

                    # Add successful episode to dataset
                    dataset.extend(episode_data[env_idx])
                    print(f"EXPERT: [Env {env_idx}] Episode {env_episode_idx[env_idx]} SUCCESS "
                            f"in {env_step_count[env_idx]} steps "
                            f"(Total: {episode_count}/{target_episodes})")
                else:
                    print(f"EXPERT: [Env {env_idx}] Episode {env_episode_idx[env_idx]} FAILED "
                            f"in {env_step_count[env_idx]} steps "
                            f"(Total: {episode_count}/{target_episodes})")
                
                # Reset episode data for this environment
                episode_data[env_idx] = []
                env_step_count[env_idx] = 0
                
                # Plan new path if we still need more episodes
                if episode_count < target_episodes:
                    a_star_path = env.env_method('plan_global_path_astar', indices=env_idx)[0]
                    
                    # Check if path planning succeeded
                    if a_star_path is None or len(a_star_path) == 0:
                        print(f"EXPERT: [Env {env_idx}] Episode {env_episode_idx[env_idx] + 1}: "
                            f"Path planning FAILED, continuing with P-controller...")
                    else:
                        print(f"EXPERT: [Env {env_idx}] Episode {env_episode_idx[env_idx] + 1}: "
                            f"Planned path with {len(a_star_path)} waypoints")
        
        # Update observations for next step
        obs = next_obs
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"EXPERT: Expert Data Collection Complete")
    print(f"{'='*60}")
    print(f"EXPERT:  Episodes collected: {episode_count}")
    print(f"EXPERT:  Successful episodes: {episode_successes}")
    print(f"EXPERT:  Success rate: {episode_successes / episode_count:.2%}")
    print(f"EXPERT:  Total transitions: {len(dataset)}")
    print(f"EXPERT:  Saving to: {Path(agent['log_dir']) / expert_cfg['dataset_name']}")
    print(f"EXPERT: {'='*60}\n")

    save_path = Path(agent["log_dir"]) / expert_cfg['dataset_name']
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    
    env.close()