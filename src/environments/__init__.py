import gymnasium as gym
import pygame

pygame.init()

gym.register("MovingTarget-v1", entry_point="environments.envs:MovingTargetEnv")
gym.register(
    "MultipleReferences-v1", entry_point="environments.envs:MultipleReferencesEnv"
)

from .wrappers import MetaObs, RewardAsInfoWrapper, RenderActionWrapper

__all__ = [
    "MetaObs",
    "RewardAsInfoWrapper",
    "RenderActionWrapper",
]
