"""
goal: turn raw expert demonstration data into a structured dataset
Pipeline:
1. Load raw data from pickle files
2. Parse raw arrays into structured entities
3. Separate numeric and lidar observations
4. Apply log transform to positional data
5. Calculate mean/std for normalization
6. Normalize numeric observations (Z-score) and lidar (max scaling)
7. Normalize actions to [-1, 1] and rewards (mean 0, std 1)
8. Recombine normalized observations into final dataset
9. Store normalization stats for future use         
"""

import pickle # for saving/loading dataset
import numpy as np
from typing import Dict, Any, List, Optional

from config import Config
from envs.entities.warehouse_observation_entity import WarehouseObservationEntity

class ExpertDataset:
    """
    Dataset wrapper for expert demonstrations.
    """

    def __init__(self, path: str = None):
        main_config = Config()
        observation = main_config.observation_dict
        self.use_laser_scan = bool(observation['enable_laser_scan'])
        self.stats: Dict[str, Any] = {}
        
        # 1. Load Raw Data
        if path is not None:
            self.raw_data = self._load(path)
        else:
            self.raw_data = []

        # 2. Prepare containers for processed data
        self.states: Optional[np.ndarray] = None
        self.next_states: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.rewards: Optional[np.ndarray] = None
        self.dones: Optional[np.ndarray] = None

        self._normalize()


    def _load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def _normalize(self):
        """Public method to trigger processing."""
        if not self.raw_data:
            print("Warning: No data to normalize.")
            return
            
        # 1. Process Observations (Populates lists)
        self._normalize_obs_egocentric()
        #self._normalize_world()
        
        # 2. Convert lists to Numpy Arrays 
        self.states = np.array(self.states, dtype=np.float32)
        self.next_states = np.array(self.next_states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        self.rewards = np.array(self.rewards, dtype=np.float32)
        self.dones = np.array(self.dones, dtype=np.float32)
        
        # 3. Process Actions
        self._normalize_actions()
        
        # 4. Final Stats
        self.stats["state_dim"] = self.states.shape[1]
        self.stats["action_dim"] = self.actions.shape[1]
        
        print(f"[ExpertDataset] Normalized {len(self.states)} transitions.")

    def _normalize_obs_egocentric(self):
        # Temporary lists
        states_list, next_states_list = [], []
        actions_list, rewards_list, dones_list = [], [], []

        MAX_DIST = 15.0
        MAX_LIDAR_DIST = 15.0
        MAX_V = 1.0
        MAX_W = 1.0

        for d in self.raw_data:
            obs_ent = WarehouseObservationEntity.from_array(np.array(d["obs"]))
            next_ent = WarehouseObservationEntity.from_array(np.array(d["next_obs"]))

            ego_obs = obs_ent.to_array_egocentric(include_laser=self.use_laser_scan)
            ego_next = next_ent.to_array_egocentric(include_laser=self.use_laser_scan)

            # 3. Normalize Base Features (Indices 0-4)
            # [dist, sin, cos, v, w, ...lidar...]
            
            # Dist: [0, 15] -> [0, 1]
            ego_obs[0] = np.clip(ego_obs[0] / MAX_DIST, 0.0, 1.0)
            ego_next[0] = np.clip(ego_next[0] / MAX_DIST, 0.0, 1.0)

            # Velocity (v): [-1, 1] -> [-1, 1]
            ego_obs[3] = np.clip(ego_obs[3] / MAX_V, -1.0, 1.0)
            ego_next[3] = np.clip(ego_next[3] / MAX_V, -1.0, 1.0)
            # Velocity (w): [-1, 1] -> [-1, 1]
            ego_obs[4] = np.clip(ego_obs[4] / MAX_W, -1.0, 1.0)
            ego_next[4] = np.clip(ego_next[4] / MAX_W, -1.0, 1.0)

            # 4. Normalize Lidar
            if len(ego_obs) > 5:
                ego_obs[5:] = np.clip(ego_obs[5:] / MAX_LIDAR_DIST, 0.0, 1.0)
                ego_next[5:] = np.clip(ego_next[5:] / MAX_LIDAR_DIST, 0.0, 1.0)

            states_list.append(ego_obs)
            next_states_list.append(ego_next)
            actions_list.append(d["action"])
            rewards_list.append(d["reward"])
            dones_list.append(d["done"])
            
        # Assign to instance 
        self.states = states_list
        self.next_states = next_states_list
        self.actions = actions_list
        self.rewards = rewards_list
        self.dones = dones_list

    # Deprecated only for testing and generating comparable results
    def _normalize_world(self):
        """
        Implementation following the log-transform + Z-score logic for navigation
        and max-scaling for lidar data.
        """
        IDX_DX = 5
        IDX_DY = 6
        
        all_numeric_states = []
        all_numeric_next_states = []
        all_lidar_states = []
        all_lidar_next_states = []
        
        all_actions = []
        all_rewards = []
        all_dones = []

        # --- 1. BUILD ARRAYS ---
        for d in self.raw_data:
            obs_ent = WarehouseObservationEntity.from_array(np.array(d["obs"]))
            next_ent = WarehouseObservationEntity.from_array(np.array(d["next_obs"]))

            # Extract numeric (8-dim) and lidar separately
            all_numeric_states.append(obs_ent.numeric_observation)
            all_numeric_next_states.append(next_ent.numeric_observation)
            
            if self.use_laser_scan and obs_ent.laser_scan is not None:
                all_lidar_states.append(obs_ent.laser_scan)
                all_lidar_next_states.append(next_ent.laser_scan)

            all_actions.append(np.array(d["action"], dtype=np.float32))
            all_rewards.append(float(d["reward"]))
            all_dones.append(float(d["done"]))

        # Convert to numpy
        np_nav_states = np.array(all_numeric_states, dtype=np.float32)
        np_nav_next_states = np.array(all_numeric_next_states, dtype=np.float32)
        
        self.actions = np.array(all_actions, dtype=np.float32)
        self.rewards = np.array(all_rewards, dtype=np.float32)
        self.dones = np.array(all_dones, dtype=np.float32)

        # --- 2. APPLY LOG TRANSFORM (Pre-processing DX/DY) ---
        for np_arr in [np_nav_states, np_nav_next_states]:
            for idx in [IDX_DX, IDX_DY]:
                np_arr[:, idx] = np.sign(np_arr[:, idx]) * np.log1p(np.abs(np_arr[:, idx]))

        # --- 3. CALCULATE STATS & NORMALIZE NAV (Z-Score) ---
        nav_mean = np_nav_states.mean(axis=0)
        nav_std = np_nav_states.std(axis=0) + 1e-3
        
        # Save stats for inference later
        self.stats["state_mean"] = nav_mean
        self.stats["state_std"] = nav_std
        self.stats["num_nav_dims"] = np_nav_states.shape[1]

        np_nav_norm = (np_nav_states - nav_mean) / nav_std
        np_nav_next_norm = (np_nav_next_states - nav_mean) / nav_std

        # --- 4. LIDAR NORMALIZATION (Max Scaling) ---
        has_lidar = len(all_lidar_states) > 0
        if has_lidar:
            np_lidar_states = np.array(all_lidar_states, dtype=np.float32)
            np_lidar_next_states = np.array(all_lidar_next_states, dtype=np.float32)
            
            lidar_max_val = np.max(np_lidar_states)
            if lidar_max_val < 1.0: lidar_max_val = 1.0
            self.stats["lidar_max"] = lidar_max_val
            
            np_lidar_norm = np_lidar_states / lidar_max_val
            np_lidar_next_norm = np_lidar_next_states / lidar_max_val
            
            # --- 5. RECOMBINE ---
            self.states = np.concatenate([np_nav_norm, np_lidar_norm], axis=1)
            self.next_states = np.concatenate([np_nav_next_norm, np_lidar_next_norm], axis=1)
        else:
            self.states = np_nav_norm
            self.next_states = np_nav_next_norm


    def _normalize_actions(self):
        """
        Normalize Actions [-1, 1] using Numpy
        """
        action_min = self.actions.min(axis=0)
        action_max = self.actions.max(axis=0)
        
        denom = action_max - action_min
        denom[denom < 1e-5] = 1.0
        
        self.actions = 2.0 * (self.actions - action_min) / denom - 1.0

        self.stats["action_min"] = action_min
        self.stats["action_max"] = action_max