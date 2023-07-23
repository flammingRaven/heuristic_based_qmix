from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv
from .SUMO_intersection_random_behaviors import MyIntersectionRandom

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["SUMO_intersection_random"] = partial(env_fn, env=MyIntersectionRandom)
