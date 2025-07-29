from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger as log
from tqdm import trange

if TYPE_CHECKING:
    from agents import BaseAgent
    from envs import BaseEnv
    from graphs import BaseGraph
    from loggers import BaseLogger


# @dataclass
# class GlobalConfig:
#     wandb_key: str = "Your wandb key"


class Simulator:
    def run(
        self,
        graph: "BaseGraph",
        agent: "BaseAgent",
        env: "BaseEnv",
        logger: "BaseLogger",
        episodes: int = None,
    ):
        episodes = trange(episodes)
        rewards = []
        for epi in episodes:
            logger.clear_logs()
            env.reset(graph=graph)
            done = False
            epi_reward = 0
            best_reward = -np.inf
            while not done:
                obs = env.demand_step(graph)
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = env.action_step(
                    graph, action
                )
                logger.log(info)
                if agent.algo.mode == "train":
                    agent.algo.model_log_reward(reward)
                epi_reward += reward
                done = terminated or truncated

            if agent.algo.mode == "train":
                agent.algo.training_step(t=epi)
                if epi_reward > best_reward:
                    best_reward = epi_reward
                    agent.algo.ckpt_update()

            rewards.append(epi_reward)
            desc = (
                f"episode: {epi + 1}, "
                f"reward: {epi_reward:.2f}, "
                f"aggregated: {np.mean(rewards):.0f} "
                f"+/- {np.std(rewards):.0f}"
            )
            episodes.set_description(desc)

            logger.log_episode(t=epi)

        logger.finish()
