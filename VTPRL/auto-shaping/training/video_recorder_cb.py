import os
import time
from typing import Any

import numpy as np
import torch
#import gymnasium.wrappers.record_video as video_recorder
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

import logging


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
        video_folder: str = None,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param video_folder: The directory in which to save the video, otherwise defaults to the TensorBoard
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._video_folder = video_folder
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

        # create video folder if it does not exist
        if self._video_folder is not None:
            os.makedirs(self._video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            t0 = time.time()

            # Use gymnasium's built-in video recording
            if self._video_folder is not None:
                video_folder = self._video_folder
                name_prefix = f"video-step-{self.n_calls}"
            else:
                import tempfile
                video_folder = tempfile.mkdtemp()
                name_prefix = "rl-video"

            # Wrap environment with RecordVideo
            from gymnasium.wrappers import RecordVideo
            wrapped_env = RecordVideo(
                self._eval_env,
                video_folder=video_folder,
                name_prefix=name_prefix,
                episode_trigger=lambda x: x < self._n_eval_episodes
            )

            rewards, lengths = evaluate_policy(
                self.model,
                wrapped_env,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            wrapped_env.close()

            t1 = time.time()
            logging.debug(f"Video recording took {t1-t0} seconds")

        return True
