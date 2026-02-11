import numpy as np
import gymnasium as gym  # <--- Replaced gym with gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

from config import Config

class EgocentricNormalizationWrapper(VecEnvWrapper):
    """
    Wraps a Vectorized Environment to transform observations from 
    absolute 'World Frame' to relative 'Egocentric Frame'.
    
    Transforms:
        Input (Raw): [x, y, yaw, v, w, dx, dy, dyaw] (8 dims)
        Output (Ego): [dist, sin(bearing), cos(bearing), v, w] (5 dims)
        
    Also handles:
        - Reward Normalization (Z-Score using dataset stats)
        - Action Un-normalization (Agent [-1,1] -> Env [Min,Max])
    """

    def __init__(
        self, 
        venv,
        stats: dict,
        normalize_obs: bool = True,
        unnormalize_actions: bool = True,
    ):
        super().__init__(venv)
        main_config = Config()
        observation = main_config.observation_dict
        self.use_laser_scan = bool(observation['enable_laser_scan'])

        self.normalize_obs = normalize_obs
        self.unnormalize_actions = unnormalize_actions
        
        if self.unnormalize_actions:
            self.action_min = np.array(stats["action_min"], dtype=np.float32)
            self.action_max = np.array(stats["action_max"], dtype=np.float32)

        # --- 2. DEFINE PHYSICAL CONSTANTS ---
        # Manual limits for normalization
        self.MAX_DIST = 15.0
        self.MAX_V = 1.0
        self.MAX_W = 1.0
        self.lidar_max = 15.0

        # --- 3. DEFINE NEW OBSERVATION SPACE ---
        self.nav_dim = 5
        total_dim = self.nav_dim
        
        if self.use_laser_scan:
            # Calculate Lidar size from the raw environment space
            # Raw shape is typically 8 + Lidar_Size
            raw_dim = venv.observation_space.shape[0]
            self.lidar_dim = raw_dim - 8 
            total_dim += self.lidar_dim


        # Use Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_dim,), 
            dtype=np.float32
        )

    def _normalize_obs(self, obs):
        """
        Transforms Raw World Obs -> Egocentric Polar Obs
        """
        if not self.normalize_obs:
            return obs
        
        # --- A. Extract Raw Variables ---
        yaw = obs[:, 2]
        v   = obs[:, 3]
        w   = obs[:, 4]
        dx  = obs[:, 5]
        dy  = obs[:, 6]

        # --- B. Compute Polar Coordinates ---
        dist = np.hypot(dx, dy)
        
        # Angle to goal (World Frame) -> Relative Bearing (Robot Frame)
        angle_to_goal = np.arctan2(dy, dx) 
        bearing = angle_to_goal - yaw
        
        # --- C. Manual Normalization ---
        # Dist: [0, MAX_DIST] -> [0, 1]
        feat_dist = np.clip(dist / self.MAX_DIST, 0.0, 1.0)
        
        # Sin/Cos: [-1, 1] -> [-1, 1]
        feat_sin = np.sin(bearing)
        feat_cos = np.cos(bearing)
        
        # Velocities: [-1, 1] -> [-1, 1]
        feat_v = np.clip(v / self.MAX_V, -1.0, 1.0)
        feat_w = np.clip(w / self.MAX_W, -1.0, 1.0)
        
        # Stack features: Shape (N, 5)
        obs_nav_norm = np.stack([feat_dist, feat_sin, feat_cos, feat_v, feat_w], axis=1)

        # --- D. Handle Lidar ---
        if self.use_laser_scan and obs.shape[1] > 8:
            obs_lidar = obs[:, 8:]
            obs_lidar_norm = np.clip(obs_lidar / self.lidar_max, 0.0, 1.0)
            return np.concatenate([obs_nav_norm, obs_lidar_norm], axis=1).astype(np.float32)
            
        return obs_nav_norm.astype(np.float32)

    def _unnormalize_actions(self, actions):
        """Scales Agent actions [-1, 1] to Environment physical limits."""
        if not self.unnormalize_actions:
            return actions

        actions = np.clip(actions, -1.0, 1.0)
        # Denormalize: y = (x + 1)/2 * (max - min) + min
        physical_actions = (actions + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min
        
        return physical_actions

    def reset(self):
        obs = self.venv.reset()
        return self._normalize_obs(obs)

    def step_async(self, actions):
        phys_actions = self._unnormalize_actions(actions)
        self.venv.step_async(phys_actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._normalize_obs(obs), rewards, dones, infos