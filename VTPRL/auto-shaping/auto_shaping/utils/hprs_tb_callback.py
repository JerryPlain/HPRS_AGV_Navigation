import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HPRSTensorboardCallback(BaseCallback):
    """
    Pull HPRS signals from VecEnv infos (filled by HPRSVecWrapper) and push to SB3 logger -> TensorBoard.
    """

    def __init__(self, log_every: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every

        # Keys to log (these must exist in infos[i])
        self.keys = [
            # success and collision
            "hprs/success", "hprs/collision",

            # phi values
            "hprs/phi_s", "hprs/phi_t", "hprs/phi_c",
            
            # delta phi values （actually shaping rewards)
            "hprs/delta_phi_s", "hprs/delta_phi_t", "hprs/delta_phi_c",
            "hprs/shaping_total",
            
            # rewards
            "hprs/base_reward", "hprs/shaped_reward_debug",
            
            # state diagnostics
            "hprs/dist_to_goal", "hprs/delta_yaw", "hprs/min_laser",
            "hprs/robot_v", "hprs/robot_omega",
        ]

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every != 0:
            return True

        infos = self.locals.get("infos", None)
        if not infos:
            return True

        # VecEnv: infos is list[dict] (one dict per env)
        for k in self.keys:
            vals = []
            for info in infos:
                v = info.get(k, None)
                if v is None:
                    continue
                # convert numpy scalars etc.
                try:
                    vals.append(float(v))
                except Exception:
                    pass

            if len(vals) == 0:
                continue

            # record mean across parallel envs
            tag = "hprs/" + k.split("/", 1)[1]   # e.g. hprs/phi_t
            self.logger.record(tag, float(np.mean(vals)))

        return True
