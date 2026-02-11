import os
import sys

# Add parent directory (agent/) to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

import argparse
import torch
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Clean imports without 'agent.' prefix
from expert.expert_dataset import ExpertDataset
import expert
sys.modules["expert"] = expert
sys.modules["expert.expert_dataset"] = sys.modules["expert.expert_dataset"]

from td3.utils.replay_buffer import ReplayBuffer
from td3.td3bc.td3_bc_agent import TD3BCAgent

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_device(device_cfg: str) -> torch.device:
    device_cfg = device_cfg.lower()
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_cfg == "cuda":
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():
    # -----------------------
    # 1. Parse command line
    # -----------------------
    parser = argparse.ArgumentParser(description="Offline TD3-BC training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    # -----------------------
    # 2. Load configuration
    # -----------------------
    cfg = load_config(args.config)
    dataset_path = cfg.get("dataset", "expert_data_hires.pkl")
    max_updates = int(cfg.get("max_updates", 100000))
    batch_size = int(cfg.get("batch_size", 256))
    alpha = float(cfg.get("alpha", 2.5))
    policy_refinement_factor = float(cfg.get("policy_refinement_factor", 1.0)) 
    learning_rate = float(cfg.get("learning_rate", 3e-4))
    gamma = float(cfg.get("gamma", 0.99))
    tau = float(cfg.get("tau", 0.005))

    models_dir = cfg.get("models_dir", "./models")
    log_dir = cfg.get("log_dir", "./runs/TD3_BC_Offline")
    save_freq = int(cfg.get("save_freq", 5000))

    device = get_device(cfg.get("device", "auto"))

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("======================================")
    print("[TD3-BC Offline] Configuration summary")
    print(f" Dataset path : {dataset_path}")
    print(f" Max updates  : {max_updates}")
    print(f" Batch size   : {batch_size}")
    print(f" Alpha        : {alpha}")
    print(f" Learning rate: {learning_rate}")
    print(f" Refinement   : {policy_refinement_factor}")
    print(f" Device       : {device}")
    print("======================================")

    # -----------------------
    # 3. Load & Process Data
    # -----------------------
    print(f"[TD3-BC Offline] Loading dataset from {dataset_path}...")
    dataset = ExpertDataset(path=dataset_path)
    
    # dataset.normalize() uses WarehouseObservationEntity internally
    # and returns flattened arrays: [Nav_ZScore, Lidar_MinMax]
    stats = dataset.stats

    state_dim = stats["state_dim"]
    action_dim = stats["action_dim"]
    max_action = 1.0                  # Always 1.0 because data is normalized to [-1, 1]

    print(f"[TD3-BC Offline] State dim  : {state_dim}")
    print(f"[TD3-BC Offline] Action dim : {action_dim}")
    print(f"[TD3-BC Offline] Max action : {max_action}")
    print(f"[TD3-BC Offline] Stats: {stats}")

    # -----------------------
    # 4. Fill Buffer
    # -----------------------
    buffer = ReplayBuffer(state_dim, action_dim, device=device)
    
    print("[TD3-BC Offline] Populating Replay Buffer...")
    for i in range(len(dataset.states)):
        buffer.add(
            dataset.states[i], 
            dataset.actions[i], 
            dataset.rewards[i], 
            dataset.next_states[i], 
            dataset.dones[i]
        )

    # -----------------------
    # 5. Initialize Agent
    # -----------------------
    # agent = TD3BCAgent(
    #         state_dim=state_dim, 
    #         action_dim=action_dim, 
    #         max_action=max_action,
    #         device=device,
    #         alpha=alpha,
    #         lr=3e-4 
    #     )
    agent = TD3BCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        alpha=alpha,
        lr=learning_rate,
    )


    writer = SummaryWriter(log_dir)

    # -----------------------
    # 6. Training loop
    # -----------------------
    print("[TD3-BC Offline] Start training...")

    for t in range(max_updates):
        
        # Policy Refinement Logic (Decay BC term or boost RL term later in training)
        current_refinement = 1.0
        if policy_refinement_factor > 1.0 and t > (max_updates / 2):
            current_refinement = policy_refinement_factor

        critic_loss, actor_loss, bc_loss = agent.train(
            replay_buffer=buffer, 
            batch_size=batch_size, 
            gamma=gamma,
            tau=tau,
            policy_refinement_factor=current_refinement
        )

        # Logging
        if (t + 1) % 100 == 0:
            writer.add_scalar("Offline/Critic_Loss", critic_loss, t)
            if actor_loss is not None:
                writer.add_scalar("Offline/Actor_Loss", actor_loss, t)
            if bc_loss is not None:
                writer.add_scalar("Offline/BC_Loss", bc_loss, t)
                # Total loss = critic + actor
                writer.add_scalar("Offline/Total_Loss", critic_loss + actor_loss, t)
            
        # Periodic Saving
        if (t + 1) % save_freq == 0:
            save_path = os.path.join(models_dir, f"td3_bc_offline_step_{t+1}")
            agent.save(save_path)
            print(f"[TD3-BC Offline] Step {t+1}: Saved model to {save_path}")
            if bc_loss is not None and actor_loss is not None:
                print(
                    f"[TD3-BC Offline] Step {t+1}: "
                    f"Critic Loss: {critic_loss:.4f}, "
                    f"Actor Loss: {actor_loss:.4f}, "
                    f"BC Loss: {bc_loss:.4f}"
                )
            else:
                print(f"[TD3-BC Offline] Step {t+1}: Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")

    # Save Final Model
    final_path = os.path.join(models_dir, "td3_bc_offline_final")
    agent.save(final_path)
    print(f"[TD3-BC Offline] Finished. Final model saved to {final_path}")
    
    writer.close()

if __name__ == "__main__":
    main()
