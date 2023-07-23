from .run import run as default_run
from .purely_evaluate import purely_eval

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["evaluate"] = purely_eval
