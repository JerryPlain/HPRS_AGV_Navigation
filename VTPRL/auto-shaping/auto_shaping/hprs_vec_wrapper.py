"""
HPRSVecWrapper: Hierarchical Potential-based Reward Shaping for VecEnv (SB3)

This module implements a vectorized version of HPRS reward shaping:
- It wraps a VecEnv (e.g. SimulatorVecEnv) and fully overrides its rewards.
- The final reward seen by the RL algorithm is:
- Sparse success reward + Potential-based shaping rewards
    
    R = R_base
        + γ * Φ_s(next) - Φ_s(prev)   # safety shaping
        + γ * Φ_t(next) - Φ_t(prev)   # target shaping
        + γ * Φ_c(next) - Φ_c(prev)   # comfort shaping

Important design choices:
- Original rewards from Unity / SimulatorVecEnv are ignored.
- Reward shaping is done entirely at the VecEnv level.
- Each parallel environment maintains its own previous extended state.

Structure:
┌───────────────────────────────┐
│        RL Algorithm           │
│   (PPO / TD3 / SAC / DQN)     │
└───────────────▲───────────────┘
                │  obs, reward, done, info
┌───────────────┴───────────────┐
│        VecEnvWrapper          │   ← HPRSVecWrapper / VecMonitor (Intercept the step output of VecEnv and replace rewards with shaped_rewards.)
│  (reward shaping / logging)   │
└───────────────▲───────────────┘
                │  VecEnv interface
┌───────────────┴───────────────┐
│           VecEnv              │   ← SimulatorVecEnv (combines multiple envs, steps them in parallel)
│  (step_async / step_wait)     │
└───────────────▲───────────────┘
                │
        ┌───────┴────────┐
        │                │
┌───────┴────────┐ ┌─────┴──────────┐
│  DummyVecEnv   │ │ SubprocVecEnv  │  ← VecEnv implementations
│(single process)│ │(multi-process) │
└───────▲────────┘ └─────▲──────────┘
        │                │
        │         ┌──────┴──────┐
        │         │   gym.Env   │
        │         └─────────────┘
        │
┌───────┴────────┐
│ gym / gymnasium│  ← WarehouseUnityEnv (bottom gym env, gives obs, done, info)
│     Env        │
└────────────────┘
"""

from typing import Optional, List, Dict, Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper # DummyVecEnv is a subclass of VecEnv

from auto_shaping.spec.reward_spec import RewardSpec, Variable, Constant
from auto_shaping.utils.utils import clip_and_norm

# mapping string operators in config.yaml to actual comparison functions
_cmp_lambdas = {
    "<": lambda x, y: x < y,
    ">": lambda x, y: x > y,
    "<=": lambda x, y: x <= y,
    ">=": lambda x, y: x >= y,
    "==": lambda x, y: abs(x - y) < 1e-6,
    "!=": lambda x, y: abs(x - y) > 1e-6,
}

# mapping string function names to actual functions
_fns = {
    "abs": np.abs,
    "exp": np.exp,
    None: lambda x: x,
}


def extend_state_warehouse(base_env, state_vec: np.ndarray, info: Dict[str, Any]):
    """
    Convert the raw observation vector from WarehouseUnityEnv
    into the extended state dictionary required by HPRS.

    This function is THE ONLY PLACE where
    - the observation format of WarehouseUnityEnv is interpreted.
    - the mapping from observation vector to HPRS variables is defined.

    - take state_vec: raw observation vector from WarehouseUnityEnv
    - return: extended state dictionary with all variables required by HPRS
    ️ Important: variable names in the returned dictionary must match those defined in RewardSpec YAML file.
    ️ Important: info dictionary from env.step() is also passed in for additional flags
    ️ Important: base_env is the actual WarehouseUnityEnv instance, can access its attributes if needed.
    ️ Note: state_vec is a 1D numpy array, its content/format depends on WarehouseUnityEnv implementation.
    ️ Note: info dictionary may contain additional episode-level flags like "collision", "success".
    ️ Note: base_env may have additional attributes like laser scan data if needed.
    ️ Note: This function must be kept in sync with the observation format of WarehouseUnityEnv.
    """
    # Robot pose, orientation, velocities from observation vector
    robot_x = float(state_vec[0])
    robot_y = float(state_vec[1])
    robot_yaw = float(state_vec[2])
    robot_v = float(state_vec[3])
    robot_omega = float(state_vec[4])

    # goal-relative deltas from observation vector
    delta_x = float(state_vec[5]) # delta x to goal
    delta_y = float(state_vec[6]) # delta y to goal
    delta_yaw = float(state_vec[7]) # delta yaw to goal
    dist_to_goal = float(np.hypot(delta_x, delta_y)) # Euclidean distance to goal
    pose_error = float(dist_to_goal + 0.5 * abs(delta_yaw)) # weighted pose error

    # Episode-level flags from info dictionary
    collision = 1.0 if info.get("collision", False) else 0.0
    success = 1.0 if info.get("success", False) else 0.0

    # Minimum distance to obstacle from laser scan, which will be used in safety shaping
    if getattr(base_env, "use_laser_scan", False): # base_env.use_laser_scan == True, then read laser scan
        laser_vec = getattr(base_env, "unity_observation", {}).get("laser_scan", None) # if there is no unity_observation, return None; if there is no laser_scan in dict, return None
        if laser_vec is not None and len(laser_vec) > 0:
            min_laser = float(np.min(laser_vec)) # take the closest obstacle distance
        else:
            min_laser = 10.0 # No obstacles detected
    else:
        min_laser = 10.0

    # Return a dictionary containing all variables required by HPRS reward shaping
    # These variable names must match those defined in the RewardSpec YAML file
    return {
        "robot_x": robot_x,
        "robot_y": robot_y,
        "robot_yaw": robot_yaw,
        "robot_v": robot_v,
        "robot_omega": robot_omega,

        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_yaw": delta_yaw,
        "dist_to_goal": dist_to_goal,
        "pose_error": pose_error,

        "collision": collision,
        "min_laser": min_laser,
        "collision": float(info.get("collision", 0.0)),
        "success": success,
    }


class HPRSVecWrapper(VecEnvWrapper):
    """
    This is the vectorized version of HPRS reward shaping.
    Key Responsibilities:
    - Wrap a VecEnv (e.g. SimulatorVecEnv).
    - Override its rewards using HPRS formula.
    - Maintain per-environment previous extended states for potential difference calculation. (γΦ(s') - Φ(s))
    """

    def __init__(
        self,
        venv: VecEnv,  # DummyVecEnv is a subclass of VecEnv
        reward_spec: RewardSpec, # The reward specification defining HPRS variables and specs
        gamma: float = 1.0,  # Discount factor for future rewards, typically 0.9–1.0
                             # Reward = base reward (target) + ΔΦ
                             # State progress ΔΦ = γ * Φ(s') - Φ(s)
                             # Φ(s): defined potential function
        shaping_scale: float = 1.0, # defualt 1, if >1, increase the impact of shaping rewards for faster learning
        debug_print: bool = True,  # Whether to print debug information during stepping
    ):
        # Initialize the VecEnvWrapper base class
        super().__init__(venv)

        # Store HPRS configuration
        self._spec: RewardSpec = reward_spec
        self._gamma = gamma
        self._shaping_scale = shaping_scale
        self.debug_print = debug_print

        # Extract variables and constants from RewardSpec
        self._variables: dict[str, Variable] = reward_spec.variables
        self._constants: dict[str, Constant] = reward_spec.constants

        # Split specs into three categories; each is a list of RequirementSpec
        self._safety_specs = [s for s in reward_spec.specs if s._operator == "ensure"] # safety specs
        self._target_specs = [s for s in reward_spec.specs if s._operator in ["achieve", "conquer"]] # target specs (is gated by safety)
        self._comfort_specs = [s for s in reward_spec.specs if s._operator == "encourage"] # comfort specs (is gated by safety and target)

        # target has 2 responsibilities: success reward + target shaping
        # allow multiple target specs and aggregate their (normalized) rewards
        # (soft gating instead of hard single-target constraint)

        """
        Each element in self._last_states corresponds to one parallel environment.
        It will store the extended state dictionary returned by extend_state_warehouse,
        which contains all variables required by HPRS reward shaping.

        - Number of environments: self.num_envs, which inherited from VecEnvWrapper
        - self._last_states: list of length self.num_envs
        
        e.g. for num_envs = 3:
        self._last_states = [
            {var1: val, var2: val, ...},  # for env 0
            {var1: val, var2: val, ...},  # for env 1
            {var1: val, var2: val, ...},  # for env 2,
        ]
        """
        # Initialize a list of length num_envs: [None, None, ...]
        self._last_states: list[Optional[dict]] = [None for _ in range(self.num_envs)]

        # Also track last actions for each environment
        self._last_actions: np.ndarray = np.zeros(
            (self.num_envs, self.action_space.shape[0]),  # Shape: (n_envs, action_dim)
            dtype=np.float32,
        )

        # Track how many environment steps have been executed since reset
        self._step_count: int = 0

    def _sanitize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Keep observations finite to avoid NaN/Inf propagation into the policy/critic.
        For Warehouse, laser scan (if present) starts at index 8; clip to a safe range.
        """
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        if obs.ndim == 2 and obs.shape[1] > 8:
            obs[:, 8:] = np.clip(obs[:, 8:], 0.0, 100.0)
        return obs

    def _sanitize_single_obs(self, obs: np.ndarray) -> np.ndarray:
        if obs is None:
            return obs
        obs = np.asarray(obs, dtype=np.float32)
        return self._sanitize_obs(obs[None, :])[0]

    # -------------------------------------------------------------------------------------------
    # Helper methods to access base envs (single WarehouseUnityEnv instances for laser scan etc.)
    # -------------------------------------------------------------------------------------------

    def _get_base_vec_env(self):
        """
        Given a wrapped VecEnv like: VecMonitor(HPRSVecWrapper(SimulatorVecEnv(...)))   
        Unwrap all VecEnvWrappers layer by layer to obtain the innermost SimulatorVecEnv.
        This allows access to envs[i], which are the actual WarehouseUnityEnv instances.
        """
        # Start from the outermost VecEnv
        base = self.venv
        while hasattr(base, "venv"): # If it's a VecEnvWrapper, unwrap one layer
            base = base.venv
        return base # Now base is the innermost VecEnv (SimulatorVecEnv)

    def _get_base_env(self, index: int):
        """
        Get the actual single environment object (WarehouseUnityEnv) at the given index.
        """
        # get the innermost VecEnv
        base_vec = self._get_base_vec_env()

        # SimulatorVecEnv has an .envs list, each is a WarehouseUnityEnv
        return base_vec.envs[index]

    # ---------------------------
    # VecEnv interface methods
    # ---------------------------
    def reset(self):
        """
        This is not for resetting the underlying environments,
        but to initialize all the "last state and last action for HPRS" for each environment before each episode.
        """
        # Call underlying VecEnv reset
        obs = self.venv.reset() # get raw observations from underlying envs; HPRS does not change obs, only rewards
        obs = self._sanitize_obs(obs)
        self._step_count = 0 # Reset step count for shaping debug print
        
        # initialize _last actions for each environment
        zero_action = np.zeros(self.action_space.shape[0], dtype=np.float32) # zero action for initialization

        # Initialize _last_states for each environment
        for i in range(self.num_envs):
            base_env = self._get_base_env(i) # get the actual WarehouseUnityEnv instance
            self._last_states[i] = extend_state_warehouse(base_env, obs[i], {}) # extend the initial observation to HPRS state
            self._last_actions[i] = zero_action # Initialize last action to zero

        # Debug print
        if self.debug_print:
            print("[HPRSVec] reset done, initialized extended states.")

        return obs # only return the original observations, not changing obs for PPO/TD3/SAC/DQN
    
    # ---------------------------
    # Core stepping logic
    # ---------------------------
    """
    In SB3 VecEnvWrapper, the stepping process is divided into two methods:
    step_async(actions): send actions to all environments
    step_wait(): wait for all environments to complete and return results

    Wrapper usually only overrides step_wait() to modify the outputs (obs, rewards, dones, infos).
    """
    def step_wait(self):
        """
        Change Reward to HPRS-shaped reward
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self._sanitize_obs(obs)
        shaped_rewards = np.zeros_like(rewards)

        # For each parallel environment, compute shaped reward
        for i in range(self.num_envs):
            base_env = self._get_base_env(i)

            if dones[i] and "terminal_observation" in infos[i]:
                # If the episode ended, use the terminal observation to calculate the final reward
                # SB3 puts the final state of the episode in infos[i]["terminal_observation"]
                final_obs = infos[i]["terminal_observation"]
                
                # We extend the state based on where the robot WAS when it finished
                next_state = extend_state_warehouse(base_env, final_obs, infos[i])
            else:
                # Standard case: use the current observation
                next_state = extend_state_warehouse(base_env, obs[i], infos[i])
            
            """
            In VecEnv, when an environment returns done=True,
            the obs returned is the initial observation after reset.
            However, for HPRS reward calculation, we need the actual final state before reset.
            """

            # when done, infos[i] contains "terminal_observation" key with the actual final state before reset
            if dones[i] and "terminal_observation" in infos[i]:
                actual_next_obs = self._sanitize_single_obs(infos[i]["terminal_observation"])
            else:
                actual_next_obs = obs[i]

            # Pass the info dictionary to the extender
            # info has episode-level flags like "collision", "success", if we don't pass info, we cannot use those flags in HPRS shaping
            next_state = extend_state_warehouse(base_env, actual_next_obs, infos[i])

            #########################################################
            # Debug and visualization info logging (TensorBoard etc.)
            #########################################################

            # compute potentials on next_state for logging
            base_r = float(self._base_reward(next_state))
            phi_s_next = float(self._safety_shaping(next_state))
            phi_t_next = float(self._target_shaping(next_state))
            phi_c_next = float(self._comfort_shaping(next_state))

            infos[i]["hprs/base_reward"] = base_r
            infos[i]["hprs/phi_s"] = phi_s_next
            infos[i]["hprs/phi_t"] = phi_t_next
            infos[i]["hprs/phi_c"] = phi_c_next

            # compute potentials on prev state and calculate delta phi for logging
            if self._last_states[i] is None:
                phi_s_prev = 0.0
                phi_t_prev = 0.0
                phi_c_prev = 0.0
            else:
                phi_s_prev = float(self._safety_shaping(self._last_states[i]))
                phi_t_prev = float(self._target_shaping(self._last_states[i]))
                phi_c_prev = float(self._comfort_shaping(self._last_states[i]))
            
            delta_phi_s = float(self._gamma * phi_s_next - phi_s_prev)
            delta_phi_t = float(self._gamma * phi_t_next - phi_t_prev)
            delta_phi_c = float(self._gamma * phi_c_next - phi_c_prev)
            
            shaping_total = (delta_phi_s + delta_phi_t + delta_phi_c) * self._shaping_scale
            
            # Optional: additional dense shaping based on distance progress (prev_dist - curr_dist)
            dist_delta_bonus = 0.0
            if self._last_states[i] is not None:
                dist_prev = float(self._last_states[i].get("dist_to_goal", 0.0))
                dist_now = float(next_state.get("dist_to_goal", dist_prev))
                dist_delta = dist_prev - dist_now # progress towards goal
                dist_w = self._constants.get("delta_dist_weight")
                if dist_w is not None:
                    dist_delta_bonus = float(dist_w.value) * dist_delta
            infos[i]["hprs/delta_dist_bonus"] = dist_delta_bonus

            infos[i]["hprs/delta_phi_s"] = delta_phi_s
            infos[i]["hprs/delta_phi_t"] = delta_phi_t
            infos[i]["hprs/delta_phi_c"] = delta_phi_c
            infos[i]["hprs/shaping_total"] = shaping_total + dist_delta_bonus

            # shaped reward debug (base + shaping)
            shaped_r = float(base_r + shaping_total + dist_delta_bonus)
            infos[i]["hprs/shaped_reward_debug"] = shaped_r # sanity check: should equal to shaped_rewards[i] computed below

            # raw extended state fields logging for debugging
            for k in ["collision","success","min_laser","dist_to_goal","delta_yaw","robot_v","robot_omega"]:
                infos[i][f"hprs/{k}"] = float(next_state[k])
                
            # ---------------------------------------------------
            # Compute shaped reward
            shaped = self._hprs_reward(
                state=self._last_states[i], 
                action=self._last_actions[i], 
                next_state=next_state, 
                done=bool(dones[i]), 
                info=infos[i], 
            )
            shaped += dist_delta_bonus

            # Optional: explicit collision penalty (in addition to safety gating)
            collision_pen = self._constants.get("collision_penalty")
            if collision_pen is not None and float(next_state.get("collision", 0.0)) >= 1.0:
                shaped -= float(collision_pen.value)

            infos[i]["hprs/shaped_reward_final"] = float(shaped)
            shaped_rewards[i] = shaped

            # Update Memory to prevent cross-episode potential difference calculation
            if dones[i]:
                # If done, we reset the memory for the next step calculation to None
                # to prevent calculating potential difference across episodes
                self._last_states[i] = None
            else:
                self._last_states[i] = next_state

        self._step_count += 1 

        if self.debug_print and (self._step_count % 50 == 0):
            j = 0
            print("[DEBUG HPRS] env0 next_state:", self._last_states[j])
            print(f"[DEBUG HPRS] env0 phi_s={infos[j].get('hprs/phi_s')}, phi_t={infos[j].get('hprs/phi_t')}, phi_c={infos[j].get('hprs/phi_c')}")
            print(
                f"[HPRSVec] step {self._step_count}: "
                f"mean raw = {float(rewards.mean()):.4f}, "
                f"mean shaped = {float(shaped_rewards.mean()):.4f}, "
            )

        return obs, shaped_rewards, dones, infos

    # ---------------------------
    # HPRS logic
    # ---------------------------

    def _base_reward(self, next_state: dict) -> float:
        # """
        # Sparse success reward: +1 if target achieved, else 0
        # """
        # reward = 0.0
        # for spec in self._target_specs:
        #     (fn, var_name), op, threshold = spec._predicate.to_tuple()
        #     cmp_f = _cmp_lambdas[op]
        #     val = _fns[fn](next_state[var_name])
        #     reward += float(cmp_f(val, threshold))
        # return reward

        """
        Sparse success reward: +1 if env reports success, else 0
        success is provided by extend_state_warehouse from info dict
        """
        return float(next_state.get("success", 0.0) >= 1.0)

    def _safety_shaping(self, state: dict) -> float:
        """
        only one ensure spec is allowed
        ️ Safety shaping is binary: 1 if all safety specs are satisfied, else 0
        ️ If any safety spec is violated, safety shaping is 0
        ️ This creates a hard safety boundary
        """
        r = 0.0
        for spec in self._safety_specs:
            (fn, var), op, th = spec._predicate.to_tuple()
            r += float(_cmp_lambdas[op](_fns[fn](state[var]), th))
        return r

    def _target_shaping(self, state: dict) -> float:
        """"
        Gated by safety
        - Safety_mask: 1 if all safety specs are satisfied, else 0
        - Target_reward is computed only when safety_mask == 1 and it is continuous between 0 and 1 using clip_and_norm for normalization
        When safety is violated, target shaping is 0
        When safety is satisfied, target shaping increases as we get closer to the target
        """
        r = 0.0
        safety_mask = 1.0
        for spec in self._safety_specs:
            (fn, var), op, th = spec._predicate.to_tuple()
            safety_mask *= float(_cmp_lambdas[op](_fns[fn](state[var]), th))

        target_vals = []
        for spec in self._target_specs:
            (fn, var), op, th = spec._predicate.to_tuple()
            val = _fns[fn](state[var])

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if op in [">", ">="]:
                target_reward = clip_and_norm(val, minv, th)
            else:
                target_reward = 1.0 - clip_and_norm(val, th, maxv)

            target_vals.append(target_reward)

        if len(target_vals) > 0:
            r += safety_mask * float(np.mean(target_vals))

        return r

    def _comfort_shaping(self, state: dict) -> float:
        """
        Gated by safety and target
        - Safety_mask: 1 if all safety specs are satisfied, else 0
        - Target_mask: 1 if all target specs are satisfied, else 0
        - Comfort_reward is computed only when both safety_mask and target_mask == 1 and it is continuous between 0 and 1 using clip_and_norm for normalization
        When safety is violated, comfort shaping is 0
        When target is not achieved, comfort shaping is 0
        When both safety and target are satisfied, comfort shaping increases as we get closer to the comfort threshold
        """
        r = 0.0

        # Safety gating
        safety_mask = 1.0
        for spec in self._safety_specs:
            (fn, var), op, th = spec._predicate.to_tuple()
            safety_mask *= float(_cmp_lambdas[op](_fns[fn](state[var]), th))

        # Target gating (soft): use mean of target rewards to avoid hard gating
        target_vals = []
        for spec in self._target_specs:
            (fn, var), op, th = spec._predicate.to_tuple()
            val = _fns[fn](state[var])

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if op in [">", ">="]:
                target_vals.append(clip_and_norm(val, minv, th))
            else:
                target_vals.append(1.0 - clip_and_norm(val, th, maxv))

        target_mask = float(np.mean(target_vals)) if target_vals else 1.0

        # Comfort shaping
        for spec in self._comfort_specs:
            (fn, var), op, th = spec._predicate.to_tuple()
            val = _fns[fn](state[var])

            minv = self._variables[var].min
            maxv = self._variables[var].max

            if op in [">", ">="]:
                comfort = clip_and_norm(val, minv, th)
            else:
                comfort = 1.0 - clip_and_norm(val, th, maxv)

            r += safety_mask * target_mask * comfort

        return r
    
    def _hprs_reward(
        self,
        state: Optional[dict],
        action: np.ndarray,
        next_state: dict,
        done: bool,
        info: dict,
    ) -> float:
        
        # Compute the base reward
        base_reward = self._base_reward(next_state)

        # if state is None or done, return base reward only
        if state is None or done:
            return base_reward
        
        # Compute shaped rewards
        safety = self._gamma * self._safety_shaping(next_state) - self._safety_shaping(state)
        target = self._gamma * self._target_shaping(next_state) - self._target_shaping(state)
        comfort = self._gamma * self._comfort_shaping(next_state) - self._comfort_shaping(state)

        total_shaping = (safety + target + comfort) * self._shaping_scale # scale shaping rewards

        return base_reward + total_shaping
