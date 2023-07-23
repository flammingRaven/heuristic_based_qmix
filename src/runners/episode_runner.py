from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0
        self.return_mean = None
        self.complete_flag_sum_test = 0.0
        self.complete_flag_sum_train = 0.0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset() if not self.args.use_curricula else self.env.reset(self.args.env_args)
        self.t = 0

    def run(self, test_mode=False, stage=0, phase=0):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # actions = self.actions_selection_curricula(test_mode=test_mode, stage=stage)
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()
            
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        # print("epsilon: ", self.mac.action_selector.epsilon)
        actions = self.actions_selection_curricula(test_mode=test_mode, phase=phase)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        if not self.test_stats:
            self.complete_flag_sum_test = 0.0
        if not self.train_stats:
            self.complete_flag_sum_train = 0.0
        # print("test_stats: ", self.test_stats)
        # print("train_stats: ", self.train_stats)
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        if test_mode:
            self.complete_flag_sum_test += float(self.env.complete_flag)
        else:
            self.complete_flag_sum_train += float(self.env.complete_flag)
        cur_stats["complete_flag"] = self.complete_flag_sum_test if test_mode else self.complete_flag_sum_train
        cur_stats["collisions"] = self.env.collision_times
        # print(self.env.complete_flag, self.env.enter_flag_list, self.env.leave_flag_list)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        self.return_mean = np.mean(cur_returns)
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def actions_selection_curricula(self, test_mode, stage=0, phase=0):
        actions = None
        if phase == 0:  # for no use of curriculum
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        elif phase == 1:  # for curriculum stage 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env - self.args.t_stage_1,
                                              test_mode=test_mode)
        elif phase == 2:  # for curriculum stage 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env - self.args.t_stage_2,
                                              test_mode=test_mode)
        return actions

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
