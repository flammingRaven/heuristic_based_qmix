import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import json
import numpy as np
import os


def correct_data_unequal_dimension(vals, vals_T):
    if len(vals) == len(vals_T):
        return vals, vals_T
    elif len(vals) - len(vals_T) == 1:
        return vals[:-1], vals_T
    elif len(vals_T) - len(vals) == 1:
        return vals, vals_T[:-1]
    else:
        print("The dimensions are not matched")
        raise RuntimeError


class DataPlotter:
    def __init__(self, save_path, load_path, tag, mode):
        self.save_path = save_path
        self.load_path = load_path
        self.tag = tag
        self.mode = mode
        self.modes = ["train", "test"]
        self.tags = ["fixed_route", "random_route"]
        assert tag in self.tags and mode in self.modes
        self.algo_labels = ["IPPO", "QMIX", "H-QMIX"] if self.tag == "fixed_route" else ["IPPO", "H-QMIX"]
        self.first_cases = ["_ippo_WITH_action_mask", "_qmix_no_action_mask", "_qmix_WITH_action_mask"]
        self.second_cases = ["_ippo_no_curriculum", "_qmix_no_curriculum"]
        self.seeds_num = 3
        self.ylim_for_ep_ret = [-500, 600] if self.tag == "fixed_route" else [-200, 600]
        # set for colors used for different algos:
        self.hqmix_color = "darkorange" #if self.tag == "fixed_route" else "darkorange"
        self.qmix_color = "violet"
        self.ippo_color = "limegreen" #if self.tag == "fixed_route" else "royalblue"
        if tag == "fixed_route":
            self.ep_ret = [[] for _ in range(len(self.first_cases))]
            self.complete_ratios = [[] for _ in range(len(self.first_cases))]
            self.episode_len = [[] for _ in range(len(self.first_cases))]
            self.episode_steps = [[] for _ in range(len(self.first_cases))]
            # self.complete_ratios_T = [[] for _ in range(len(self.first_cases))]
            # self.episode_len_T = [[] for _ in range(len(self.first_cases))]
        elif tag == "random_route":
            self.ep_ret = [[] for _ in range(len(self.second_cases))]
            self.complete_ratios = [[] for _ in range(len(self.second_cases))]
            self.episode_len = [[] for _ in range(len(self.second_cases))]
            self.episode_steps = [[] for _ in range(len(self.second_cases))]
            # self.complete_ratios_T = [[] for _ in range(len(self.first_cases))]
            # self.episode_len_T = [[] for _ in range(len(self.first_cases))]

    def plot(self):
        self._load_data()
        episode_steps, ep_rets, complete_ratios, episode_lens = self._fit_data_using_pchip_interpolator()
        # plot episode return
        self._plot_shaded_curve_for_single_metric_multi_algos(episode_steps, ep_rets, ylabel="Episode Return")
        # plot complete ratios
        self._plot_shaded_curve_for_single_metric_multi_algos(episode_steps, complete_ratios, ylabel="Success Rate")
        # plot episode length
        self._plot_shaded_curve_for_single_metric_multi_algos(episode_steps, episode_lens, ylabel="Episode Length")

    def _load_data(self):
        ############# exp1: fixed route data ##############
        load_dir_paths = []
        cases = self.first_cases if self.tag == "fixed_route" else self.second_cases
        for i, c in enumerate(cases):
            path = self.load_path + self.tag + c
            dir_path = [path for _ in range(len(os.listdir(path)))]
            for j, name in enumerate(os.listdir(path)):
                dir_path[j] += "/" + name + "/info.json"
            load_dir_paths.append(dir_path)

        for i, paths in enumerate(load_dir_paths):  # for different algorithms
            for path in paths:  # for different random seeds
                with open(path) as f:
                    data = json.load(f)
                    ##### for complete ratios: #######
                    complete_flag_mean = data["test_complete_flag_mean"] if self.mode == "test" else \
                        data["complete_flag_mean"]
                    complete_flag_mean_T = data["test_complete_flag_mean_T"] if self.mode == "test" else \
                        data["complete_flag_mean_T"]
                    complete_flag_mean, complete_flag_mean_T = \
                        correct_data_unequal_dimension(complete_flag_mean, complete_flag_mean_T)
                    complete_flag_mean, complete_flag_mean_T = np.array(complete_flag_mean), np.array(complete_flag_mean_T)
                    self.episode_steps[i].append(complete_flag_mean_T)  # total episode timesteps
                    self.complete_ratios[i].append(complete_flag_mean)

                    ##### for episodic return #######
                    return_mean = []
                    data_mean_temp = data["return_mean"] if self.mode == "test" else data["test_return_mean"]
                    for item in data_mean_temp:
                        return_mean.append(item["value"])
                    # return_mean_T = data["test_return_mean_T"] if self.mode == "test" else data["return_mean_T"]
                    return_mean, complete_flag_mean_T = \
                        correct_data_unequal_dimension(return_mean, complete_flag_mean_T)
                    self.ep_ret[i].append(np.array(return_mean))

                    ####### for episode length #########
                    episode_length = data["test_ep_length_mean"] if self.mode == "test" else data["ep_length_mean"]
                    episode_length, complete_flag_mean_T = \
                        correct_data_unequal_dimension(episode_length, complete_flag_mean_T)
                    # episode_length_T = data["test_ep_length_mean_T"] if self.mode == "test" else data["ep_length_mean_T"]
                    self.episode_len[i].append(np.array(episode_length))
        self.ep_ret = np.array(self.ep_ret)
        # self.ep_ret = self._convert_nested_list_to_ndarray(self.ep_ret)
        # print(self.ep_ret)
        # self.complete_ratios = np.array(self.complete_ratios)
        self.episode_len = np.array(self.episode_len)
        self.episode_steps = np.array(self.episode_steps)

    # def _convert_nested_list_to_ndarray(self, vals):
    #     vs = []
    #     for val in vals:
    #         for v in val:
    #             v = np.array(v)
    #             vs.append(v)
    #     vs = np.array(vs)
    #     print(vs[0][0])
    #     return vs

    def _fit_data_using_pchip_interpolator(self):
        ep_rets = []
        complete_ratios = []
        episode_lens = []
        episode_steps = []

        # data dimension: [algo_nums, seed_nums]
        for i in range(len(self.first_cases) if self.tag == "fixed_route" else len(self.second_cases)):  # for each algo
            base_x_axis = self.episode_steps[i][0]  # chooose the base T for all seeds
            episode_steps.append(base_x_axis)
            ep_ret_new, complete_ratios_new, ep_len_new = self._fit_and_compress_data_for_multiple_seeds(i, base_x_axis)
            ep_ret_max, ep_ret_min, ep_ret_mean = np.max(ep_ret_new, axis=0), \
                                                  np.min(ep_ret_new, axis=0), \
                                                  np.mean(ep_ret_new, axis=0)
            complete_ratios_max, complete_ratios_min, complete_ratios_mean = np.max(complete_ratios_new, axis=0), \
                                                                             np.min(complete_ratios_new, axis=0), \
                                                                             np.mean(complete_ratios_new, axis=0)
            ep_len_max, ep_len_min, ep_len_mean = np.max(ep_len_new, axis=0), np.min(ep_len_new, axis=0), \
                                                  np.mean(ep_len_new, axis=0)
            ep_rets.append([ep_ret_max, ep_ret_min, ep_ret_mean])
            complete_ratios.append([complete_ratios_max, complete_ratios_min, complete_ratios_mean])
            episode_lens.append([ep_len_max, ep_len_min, ep_len_mean])
        ep_rets = np.array(ep_rets)
        complete_ratios = np.array(complete_ratios)
        episode_lens = np.array(episode_lens)
        return episode_steps, ep_rets, complete_ratios, episode_lens

    def _fit_and_compress_data_for_multiple_seeds(self, algo_id, base_axis):
        ep_ret_interpolators = []
        complete_ratios_interpolators = []
        ep_len_interpolators = []
        ep_ret_new = []
        complete_ratios_new = []
        ep_len_new = []
        ep_ret_interpolator = None
        complete_ratio_interpolator = None
        ep_len_interpolator = None
        for j in range(self.seeds_num):  # for seed nums
            ep_ret_interpolator = PchipInterpolator(self.episode_steps[algo_id][j], self.ep_ret[algo_id][j])
            complete_ratio_interpolator = PchipInterpolator(self.episode_steps[algo_id][j],
                                                            self.complete_ratios[algo_id][j])
            ep_len_interpolator = PchipInterpolator(self.episode_steps[algo_id][j], self.episode_len[algo_id][j])
            ep_ret_new.append(ep_ret_interpolator(base_axis))
            complete_ratios_new.append(complete_ratio_interpolator(base_axis))
            ep_len_new.append(ep_len_interpolator(base_axis))
        ep_ret_new = np.array(ep_ret_new)
        complete_ratios_new = np.array(complete_ratios_new)
        ep_len_new = np.array(ep_len_new)
        return ep_ret_new, complete_ratios_new, ep_len_new

    def _plot_shaded_curve_for_single_metric_multi_algos(self, xs, vals, ylabel, color="b", shaded_ratio=.5):
        # vals include: max values, min values, mean values
        # examples of `ylabel`: "Episode Return", "Complete Ratio", "Episode Length"
        plt.figure(0)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plot ep_ret:
        for i, val in enumerate(vals):  # for different algo
            if self.algo_labels[i] == "H-QMIX":
                color = self.hqmix_color
            elif self.algo_labels[i] == "QMIX":
                color = self.qmix_color
            elif self.algo_labels[i] == "IPPO":
                color = self.ippo_color
            max_vals = val[0]
            min_vals = val[1]
            mean_vals = val[2]
            ax.fill_between(xs[i], max_vals, min_vals, color=color, alpha=shaded_ratio, linewidth=0)
            ax.plot(xs[i], mean_vals, color=color, linewidth=1, label=self.algo_labels[i])
            plt.xlabel('Time Step (s)')
            plt.ylabel(ylabel)
            plt.legend()
            if ylabel == "Episode Return":
                plt.ylim(self.ylim_for_ep_ret)
        plt.grid()
        str1 = ylabel.split(" ")[0]
        str2 = ylabel.split(" ")[1]
        save_path = self.save_path + "/" + str1 + "_" + str2 + "_" + self.tag + "_comparison_" + self.mode + ".pdf"
        plt.savefig(save_path, format='pdf', dpi=500, bbox_inches='tight')
        plt.show()
        plt.close()
        # plot ep_len:


class ModelPlotter:
    def __init__(self, args, runner, save_path, hint="vel"):
        # hint: vel, fuel, accel, dist
        # save_path: absolute path
        self.hints = ["vel", "fuel", "accel", "dist"]
        self.scenarios = ["fixed_route", "random_route"]
        if hint not in self.hints:
            print("Input information is erroneous!")
            return
        self.args = args
        self.comm_lag = args.env_args["comm_lag"]
        self.hint = hint
        self.path = args.checkpoint_path
        self.algo_scenario_id = self.path.split("/")[-4]
        ## some fixed args
        self.max_vel = 15
        self.incre = args.env_args["time_step"]
        self.enter_time_min = runner.enter_time_min
        self.leave_time_max = runner.leave_time_max
        if not self.comm_lag:
            self.save_path = save_path + "/" + self.algo_scenario_id + "_"
        else:
            self.save_path = save_path + "/" + "comm_lag_" + self.algo_scenario_id + "_"
        self.lane_length = runner.env.lane_length
        self.len_intersection = runner.env.len_intersection

    def plot(self, values):
        plt.figure(0)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plt.ylim([0, 105])
        plt.cla()
        max_len = 0
        save_path = None
        for i in range(self.args.n_agents):
            x_axis = [self.incre * i for i in range(len(values[i]))]
            plt.plot(x_axis, values[i], label='veh' + str(i))
            if len(values[i]) > max_len:
                max_len = len(values[i])
            # plt.legend(['q_tot', 'reward'])
        x_axis = [self.incre * i for i in range(max_len)]
        plt.xlabel('Time Step (s)')
        plt.axvspan(self.enter_time_min, self.leave_time_max, facecolor='g', alpha=0.1, **dict())
        # todo: use design pattern for better extension if extra values are needed to plot
        if self.hint == "vel":
            # plt.axvspan(self.enter_time_min, self.leave_time_max, facecolor='g', alpha=0.1, **dict())
            print('safe speed: ', self.args.env_args["max_speed"], max_len)
            max_vel = [self.max_vel for _ in range(max_len)]
            plt.plot(x_axis, max_vel, ':', label='max_speed')
            plt.ylabel('Speed ($m/s$)')
            save_path = self.save_path + 'model_vel.pdf'

        elif self.hint == "accel":
            plt.ylabel('Acceleration ($m/s^2$)')
            save_path = self.save_path + 'model_accel.pdf'

        elif self.hint == "dist":
            intersection_start = [self.lane_length for _ in range(max_len)]
            intersection_end = [self.lane_length + self.len_intersection for _ in
                                range(max_len)]
            plt.plot(x_axis, intersection_start, ':', label='Start of the Zone')
            plt.plot(x_axis, intersection_end, ':', label='End of the Zone')
            plt.ylabel('Travelled Distance ($m$)')
            save_path = self.save_path + 'model_distance.pdf'

        elif self.hint == "fuel":
            # plt.axvspan(self.enter_time_min, self.leave_time_max, facecolor='g', alpha=0.1, **dict())
            plt.ylabel('Fuel Consumption ($mg/s$)')
            save_path = self.save_path + 'model_fuel.pdf'

        plt.legend()
        plt.grid()

        plt.savefig(save_path, format='pdf', dpi=500, bbox_inches='tight')
        fig.subplots_adjust(right=0.75)
        # np.save(self.save_path + '/q_tot_mean', q_total_mean_training)
        # np.save(self.save_path + '/rew_mean', rew_mean_training)
        plt.close()
        # plt.show()
