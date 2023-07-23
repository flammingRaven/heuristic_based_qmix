REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_eval_mode import EpisodeRunnerEvalMode
REGISTRY["episode_eval"] = EpisodeRunnerEvalMode