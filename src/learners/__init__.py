from .q_learner import QLearner
from .ppo_learner import PPOLearner
from .nq_learner import NQLearner
from .nq_learner_lr_decay import NQLearnerLRDecay

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["nq_learner_lr_decay"] = NQLearnerLRDecay
