from simulator_vec_env import SimulatorVecEnv
from envs.warehouse_unity_env import WarehouseUnityEnv
from auto_shaping.hprs_vec_wrapper import HPRSVecWrapper
from auto_shaping.spec.reward_spec import RewardSpec
from stable_baselines3.common.vec_env import VecMonitor

def get_env(agent, root, sim, gym_cfg, manip_env, warehouse_env, observation, 
            reward_dict, log_dir, hprs_spec_file, 
            additional_wrapper=None, wrapper_kwargs=None):
    """
    Create and return a vectorized warehouse environment.
    
    Args:
        ... (existing args) ...
        additional_wrapper: Optional wrapper class to apply after HPRS but before VecMonitor.
                           E.g., a normalization wrapper or OfflineStatsWrapper
        wrapper_kwargs: Dict of kwargs to pass to the additional_wrapper constructor
    
    Returns:
        Wrapped vectorized environment
    """
    env_key = gym_cfg['env_key']

    # -----------------------------------------------
    # Create single env function, id used for seeding
    # -----------------------------------------------
    def create_env(id=0):
        if env_key == 'warehouse_unity_env':
            env = WarehouseUnityEnv(
                max_time_steps=gym_cfg['max_time_step'],
                env_id=id,
                gym_config=gym_cfg['warehouse_gym_environment'],
                warehouse_config=warehouse_env,
                observation_config=observation,
            )

        env.seed((id * 150) + (id + 11))
        return env

    num_envs = gym_cfg['num_envs']
    env = [create_env for i in range(num_envs)]

    # Instantiate the vectorized environment wrapper
    env = SimulatorVecEnv(env, agent, root, sim, gym_cfg, manip_env, reward_dict=reward_dict)

    # -----------------------------------------------
    # Apply HPRSVecWrapper if in training or evaluation mode
    # -----------------------------------------------
    print("simulation_mode:", agent.get('simulation_mode'))

    # If the agent is in training or evaluation mode, wrap with HPRS for reward shaping
    if agent.get('simulation_mode') in ['train', 'evaluate']:
        spec_file = hprs_spec_file
        reward_spec = RewardSpec.from_yaml(spec_file)
        print("[HPRS] Loaded spec:", spec_file) 
        env = HPRSVecWrapper(env, reward_spec, gamma=1.0, debug_print=False)

    # -----------------------------------------------
    # Apply additional wrapper if provided (e.g., normalization)
    # -----------------------------------------------
    if additional_wrapper is not None:
        kwargs = wrapper_kwargs or {}
        env = additional_wrapper(env, **kwargs)
        print(f"[get_env] Applied additional wrapper: {additional_wrapper.__name__}")

    # Add outer monitoring wrapper
    env = VecMonitor(env, log_dir, info_keywords=("success", "collision", "timeout"))
    return env
