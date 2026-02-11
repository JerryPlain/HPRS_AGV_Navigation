from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class WarehouseObservationEntity:
    """
    Structured observation for warehouse robot navigation.
    
    Base observation (8 dims):
        - robot_x, robot_y, robot_yaw: Robot pose in world frame
        - robot_v, robot_omega: Robot velocities (linear, angular)
        - delta_x, delta_y, delta_yaw: Relative position/orientation to goal
    
    Optional:
        - laser_scan: Laser range measurements (if use_laser_scan=True)
    """

    # Robot state
    robot_x: float
    robot_y: float
    robot_yaw: float
    robot_v: float
    robot_omega: float
    
    # Goal-relative state
    delta_x: float
    delta_y: float
    delta_yaw: float
    
    # Optional sensor data
    laser_scan: Optional[np.ndarray] = None

    # --- CONSTANTS FOR VECTORIZED ACCESS ---
    IDX_X: int = 0
    IDX_Y: int = 1
    IDX_YAW: int = 2
    IDX_V: int = 3
    IDX_W: int = 4
    IDX_DX: int = 5
    IDX_DY: int = 6
    IDX_DYAW: int = 7
    
    # The size of the numeric part
    NUMERIC_DIM: int = 8
    
    @classmethod
    def from_array(cls, obs_array: np.ndarray) -> "WarehouseObservationEntity":
        """
        Create observation from raw numpy array.
        
        Args:
            obs_array: Observation array from environment
                      Shape: (8,) for base only, or (8+N,) with laser
        
        Returns:
            WarehouseObservationEntity instance
        """
        # Extract base 8 observations
        # get numeric values
        robot_x = float(obs_array[0])
        robot_y = float(obs_array[1])
        robot_yaw = float(obs_array[2])
        robot_v = float(obs_array[3])
        robot_omega = float(obs_array[4])
        delta_x = float(obs_array[5])
        delta_y = float(obs_array[6])
        delta_yaw = float(obs_array[7])
        
        # if more than 8 dims, assume laser scan follows, default None
        laser_scan = None
        if len(obs_array) > 8:
            laser_scan = obs_array[8:].astype(np.float32)
        
        return cls(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
            robot_v=robot_v,
            robot_omega=robot_omega,
            delta_x=delta_x,
            delta_y=delta_y,
            delta_yaw=delta_yaw,
            laser_scan=laser_scan
        )
    
    # Convert back to numpy array for vectorized envs
    def to_array(self, include_laser: bool = True) -> np.ndarray:
        """
        Convert back to numpy array format.
        
        Args:
            include_laser: Whether to include laser scan data
        
        Returns:
            Numpy array matching environment observation format
        """
        base = np.array([
            self.robot_x,
            self.robot_y,
            self.robot_yaw,
            self.robot_v,
            self.robot_omega,
            self.delta_x,
            self.delta_y,
            self.delta_yaw
        ], dtype=np.float32)
        
        if include_laser and self.laser_scan is not None:
            return np.concatenate([base, self.laser_scan])
        
        return base

    def to_array_egocentric(self, include_laser: bool = True) -> np.ndarray:
        """
        Calculates polar observation for a robot where:
        0 = East, 1.57 = North, 3.14 = West.
        
        Base Egocentric (5 dims): [dist, sin(bearing), cos(bearing), v, w]
        
        Args:
            include_laser: Whether to append lidar data to the end of the 5-dim vector.
            
        Returns: 
            Numpy array of shape (5,) or (5 + num_laser_points,)
        """
        # 1. Distance (Euclidean)
        dist = np.hypot(self.delta_x, self.delta_y)
        
        # 2. Absolute Angle to Goal (World Frame)
        angle_to_goal = np.arctan2(self.delta_y, self.delta_x)
        
        # 3. Relative Bearing
        diff = angle_to_goal - self.robot_yaw
        sin_b = np.sin(diff)
        cos_b = np.cos(diff)
        
        # 4. Velocities
        v = self.robot_v
        w = self.robot_omega
        
        base = np.array([dist, sin_b, cos_b, v, w], dtype=np.float32)
        
        if include_laser and self.laser_scan is not None:
            return np.concatenate([base, self.laser_scan])
            
        return base
    
    @property
    def robot_position(self) -> np.ndarray:
        """Robot position as [x, y]."""
        return np.array([self.robot_x, self.robot_y])
    
    @property
    def target_position(self) -> np.ndarray:
        """Absolute target position as [x, y]."""
        return np.array([
            self.robot_x + self.delta_x,
            self.robot_y + self.delta_y
        ])
    
    @property
    def target_yaw(self) -> float:
        """Absolute target yaw."""
        return self.robot_yaw + self.delta_yaw
    
    @property
    def distance_to_goal(self) -> float:
        """Euclidean distance to goal."""
        return float(np.hypot(self.delta_x, self.delta_y))

    @property
    def numeric_observation(self) -> np.ndarray:
        """Numeric observation as numpy array."""
        return np.array([
            self.robot_x,
            self.robot_y,
            self.robot_yaw,
            self.robot_v,
            self.robot_omega,
            self.delta_x,
            self.delta_y,
            self.delta_yaw
        ], dtype=np.float32)

    @property
    def laser_scan_observation(self) -> Optional[np.ndarray]:
        """Laser scan observation as numpy array."""
        if self.laser_scan is not None:
            return self.laser_scan
        return None
    
    @property
    def min_laser_distance(self) -> Optional[float]:
        """Minimum laser scan distance (closest obstacle)."""
        if self.laser_scan is not None:
            return float(np.min(self.laser_scan))
        return None
    
    def __repr__(self) -> str:
        laser_info = f", laser: {len(self.laser_scan)} points" if self.laser_scan is not None else ""
        return (f"WarehouseObs(pos=[{self.robot_x:.2f}, {self.robot_y:.2f}], "
                f"yaw={self.robot_yaw:.2f}, v={self.robot_v:.2f}, "
                f"dist_to_goal={self.distance_to_goal:.2f}{laser_info})")