import os
import sys
import time
from collections import deque
from collections import OrderedDict

import numpy as np
import traci
from gym.spaces import Box, Discrete
from sumolib import checkBinary

from envs.multiagentenv import MultiAgentEnv


class MyIntersectionRandom(MultiAgentEnv):
    # env of intersection scenario with CAVs of random behaviors.
    def __init__(self, **kwargs):
        self.cfg_dir = "/home/ghz/PycharmProjects/heuristic_based_qmix/src/envs/SUMO_intersection_random_behaviors/main.sumocfg"
        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")  # get the path of tools containing traci
            sys.path.append(tools)
        else:
            sys.exit("Please declare environment variable SUMO_HOME")

        self.render = kwargs["render"]
        if not self.render:
            self.sumoBinary = checkBinary("sumo")
        else:
            self.sumoBinary = checkBinary("sumo-gui")
        self.sumoCmd = [self.sumoBinary, '-c', self.cfg_dir, '--collision.check-junctions']
        self.reloadCmd = ["-c", self.cfg_dir]
        # ----------------------------------------------------------#
        self.use_multiprocessor = kwargs["multiprocess"]
        # ----------------------------------------------------------#
        self.seed = kwargs['seed']
        self.random_behavior = kwargs["random_behavior"]
        self.shared_reward = True
        self.have_own_obs = True
        self.discrete = kwargs["discrete"]
        self.comm_lag = kwargs["comm_lag"]
        self.comm_lag_curricula = kwargs["comm_lag_curricula"]
        self.env_infos = {"complete_flag": float(False), "collisions": 0.0}
        self.total_agents = kwargs['total_agents']
        self.n_agents = kwargs["n_agents"]
        self.CAVs_id_list = kwargs["CAVs_id_list"]
        self.min_reward = kwargs["min_reward"]
        self.max_reward = kwargs["max_reward"]
        self.use_ppo_like_algo = kwargs["use_ppo"]
        # --------------- communication lag related variables -----------------#
        # obs: (x, y, v, safe_distance, waiting_time, enter_flag, leave_flag)
        self.obs_temp = {"pos": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "speed": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "min_dist": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "wt": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "enter_flag": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "leave_flag": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         }

        self.routes_random_all = ["route_WE", "route_WN", "route_WS",
                                  "route_EW", "route_ES", "route_EN",
                                  "route_NE", "route_NS", "route_NW",
                                  "route_SW", "route_SN", "route_SE"]
        self.routes_fixed = {"0": "route_WS", "1": "route_WE",
                             "6": "route_SW", "7": "route_SN",
                             "3": "route_ES", "2": "route_EW",
                             "4": "route_NE", "5": "route_NS"}
        # self.same_directions = {"W": ("0", "1"), "E": ("2", "3"), "N": ("4", "5"), "S": ("6", "7")}
        self.same_directions = {"0": "1", "1": "0",
                                "2": "3", "3": "2",
                                "4": "5", "5": "4",
                                "6": "7", "7": "6"}
        # The first element is the major route
        self.routes_of_vehicles = {"0": ["route_WE", "route_WS"], "1": ["route_WN", "route_WE"],
                                   "6": ["route_SW", "route_SN"], "7": ["route_SN", "route_SE"],
                                   "3": ["route_ES", "route_EW"], "2": ["route_EW", "route_EN"],
                                   "4": ["route_NE", "route_NS"], "5": ["route_NS", "route_NW"]}
        self.intention_dim = 2
        self.intention_probs = kwargs["intention_probs"]
        self.veh_intentions = OrderedDict({})
        self.departSpeed = kwargs["depart_speed"]
        self.type_name = "VehicleA"  # "vehicle.tesla.model3"
        self.vehicle_spawn_infos = {"depart": "now", "departPos": "30.0", "departSpeed": str(self.departSpeed),
                                    "departLane": "best"}
        self.CAVs_departLane = {"0": "L6_0", "1": "L6_1",
                                "2": "L2_0", "3": "L2_1",
                                "4": "L0_1", "5": "L0_0",
                                "6": "L4_1", "7": "L4_0"}
        # self.virtual_signal_pair = {1: ("L0_0", "L4_0"), 2: ("L0_1", "L4_1"),
        #                             3: ("L6_0", "L2_0"), 4: ("L6_1", "L2_1")}
        self.virtual_signal_veh_pair = OrderedDict({0: ("5", "7"), 1: ("4", "6"),
                                        2: ("0", "2"), 3: ("1", "3")})
        self.virtual_signal_veh_pair_general = OrderedDict({0: ("route_NS", "route_NW", "route_SN", "route_SE"),
                                                            1: ("route_SW", "route_NE"),
                                                            2: ("route_WE", "route_WS", "route_EW", "route_EN"),
                                                            3: ("route_WN", "route_ES")})
        self.selected_routes_of_vehicles = OrderedDict({vehID: [] for vehID in self.CAVs_id_list})
        # self.virtual_signal_veh_pair = OrderedDict({0: ("6", "7"), 1: ("1", "0"),
        #                                 2: ("5", "4"), 3: ("2", "3")}) # all-lanes mode
        # self.virtual_signal_veh_pair = OrderedDict({0: ("4", "6"), 1: ("5", "7"),
        #                                             2: ("1", "3"), 3: ("0", "2")})
        self.signal_cycle_length = kwargs["signal_cycle_length"]
        self.use_virtual_signal = kwargs["use_virtual_signal"]
        self.time_step = kwargs["time_step"]  # 0.1
        self.decision_freq = kwargs["decision_freq"]
        self.signal_cycle_num = self.signal_cycle_length
        self.run_curricula = kwargs["run_curricula"]
        self.phase = None
        if self.run_curricula:
            self.phase = kwargs["phase"]
        # self.action_discretization = kwargs["action_discretization"]
        self.full_id_list = tuple([str(i) for i in range(self.n_agents)])
        self.max_speed = 15
        self.max_lane_length = 99
        self.render_step_time = kwargs["time_step_for_render"]
        self.time_step_for_render = self.render_step_time if self.render else 0.0
        # self.accel_dim = 2
        # self.decel_dim = self.accel_dim
        # self.action_dim_per_veh = 3  # accelerate, keep constant speed, decelerate
        self.action_pattern = kwargs["action_pattern"]
        self.accel_res = kwargs["acceleration_resolution_pattern_" + str(self.action_pattern)]
        self.action_dim_per_veh = 2 * len(self.accel_res) + 1  # accelerate, keep constant speed, decelerate
        self.accel_step = 5
        self.ways_num = 4

        self.lane_ids = ["L0_0", "L0_1", "L1_0", "L1_1", "L2_0", "L2_1", "L3_0", "L3_1",
                         "L4_0", "L4_1", "L5_0", "L5_1", "L6_0", "L6_1", "L7_0", "L7_1", ]
        self.lane_ids_follow_vehID = ["L6_1", "L6_0", "L4_1", "L4_0", "L2_1", "L2_0", "L0_1", "L0_0"]
        self.junction_name = 'J1'
        # --------episode_limit: -------#
        self.episode_limit = kwargs["episode_limit"]
        # ------------------------------#
        self.waiting_steps_threshold = int(self.episode_limit / 2)
        self.safe_dist = 5.0  # greater means safer
        self.caution_dist = 2 * self.safe_dist
        self.v_threshold = 2 / self.max_speed
        self.vehicles_through = 0
        self.complete_flag = False
        # self.lane_ids = traci.lane.getIDlist()[self.ways_num*2:]

        # ----------------------------------------------------------------------#
        self.step_num = 0 #kwargs["time_step"]
        self.episode_num = 0
        self.done = False
        self.collision_times = 0
        # self.enter_flag_list_tmp = [False for _ in range(self.n_agents)]
        # self.leave_flag_list_tmp = [False for _ in range(self.n_agents)]
        # self.enter_flag_list = [False for _ in range(self.n_agents)]
        # self.leave_flag_list = [False for _ in range(self.n_agents)]
        self.enter_flag_dict_tmp = {str(t): False for t in range(self.total_agents)}
        self.leave_flag_dict_tmp = {str(t): False for t in range(self.total_agents)}
        self.enter_flag_dict = {str(t): False for t in range(self.total_agents)}
        self.leave_flag_dict = {str(t): False for t in range(self.total_agents)}
        # ======================================== action space definition ===========================================#
        if self.discrete:
            self.action_space = [Discrete(self.action_dim_per_veh) for _ in range(self.n_agents)]
        else:
            self.action_space = Box(
                low=np.array([0 for _ in range(self.n_agents)]),
                high=np.array([self.max_speed for _ in range(self.n_agents)]),
                dtype=np.float64
            )  # action_space.sample

        # ================================== observation space definition: (x,y,v) for each agent ======================================#
        self.pos_related_dim = 2
        self.vel_end_idx = 3
        self.safe_dist_idx = 4
        self.wt_dim_idx = 5
        self.enter_flag_idx = 6
        self.leave_flag_idx = 7
        self.extra_obs_dim = 3
        self.obs_dim_per_veh = self.leave_flag_idx + self.extra_obs_dim
        # self.obs_dim_per_veh = self.leave_flag_idx

        self.observation_space = Box(
            low=np.array([[0 for _ in range(self.obs_dim_per_veh)] for _ in range(self.total_agents)]).reshape(
                self.obs_dim_per_veh * self.total_agents),
            high=np.array(
                [[float("inf"), float("inf"), self.max_speed, float("inf"), float("inf"), 1.0, 1.0, 0.0, 0.0, 0.0] for _
                 in range(self.total_agents)]).reshape(
                self.obs_dim_per_veh * self.total_agents),
            dtype=np.float64
        )

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.total_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        # stats = {
        #     "complete_flag": self.complete_flag,
        #     "collisions": self.collision_times
        # }
        stats = {}
        return stats

    def step(self, actions):
        """ Returns reward, terminated, info """
        # wait all vehicles
        self.curr_ID_list = traci.vehicle.getIDList()
        # decide if the vehicle has entered or left the intersection
        self._set_enter_and_leave_flags()
        if self.use_ppo_like_algo:
            actions = actions.cpu().numpy() if self.use_multiprocessor else actions # for ippo testing
        else:
            actions = actions.cpu().numpy() if not self.use_multiprocessor else actions  # comment out for parallel runner
        self._apply_actions(actions)
        traci.simulationStep(self.time_step * self.step_num) # self.time_step * self.step_num
        # traci.simulationStep()
        self.step_num += 1
        time.sleep(self.time_step_for_render)  # can be commented out when there is no need for gui
        # update the comm-related information
        self._update_info()
        next_observations = self.get_obs()
        rewards = self._get_shared_reward(next_observations) if self.shared_reward else self._get_rewards(
            next_observations)
        # rewards = np.clip(rewards, self.min_reward, self.max_reward)
        # print(rewards)
        done = self._get_done()
        # complete_flag = self._if_complete_task()
        return rewards, done, self.env_infos

    def _apply_actions(self, actions):
        # action is either list or np.array, if discrete, its dimension is [n_agents, action_dim]
        # assert isinstance(actions, list) or isinstance(actions, np.ndarray)
        # curr_ID_list = traci.vehicle.getIDList()
        if self.discrete:  # discrete action space
            for id_idx, id in enumerate(self.CAVs_id_list):
                if id not in self.curr_ID_list:
                    continue
                # id_idx = self.CAVs_id_list.index(id)
                traci.vehicle.setLaneChangeMode(id, 512)
                traci.vehicle.setSpeedMode(id, 32)
                curr_veh_speed = traci.vehicle.getSpeed(id)
                # speed = self._derive_speed_from_discrete_actions(id, actions)
                # accel = self._derive_accel_from_discrete_actions(id, actions) ##### serve for test_env
                # accel = self._derive_accel_from_chosen_actions_idxes(id_idx, actions[0]) # for IPPO testing
                accel = self._derive_accel_from_chosen_actions_idxes(id_idx, actions)  ##### serve for QMIX main algo
                speed = np.clip(curr_veh_speed + accel * self.time_step, a_min=0, a_max=self.max_speed)
                traci.vehicle.setSpeed(vehID=id, speed=speed)
        else:  # continuous action space
            for id_idx, id in enumerate(self.curr_ID_list):
                # id_idx = self.CAVs_id_list.index(id)
                # todo: handle the case when id is not in the current vehicle lists
                traci.vehicle.setLaneChangeMode(id, 512)
                traci.vehicle.setSpeedMode(id, 32)
                traci.vehicle.setSpeed(vehID=id, speed=actions[id_idx])

    def _update_info(self):
        # obs: (x, y, v, safe_distance, waiting_time, enter_flag, leave_flag)
        # self.obs_temp = {"pos": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "speed": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "min_dist": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "wt": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "enter_flag": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "leave_flag": [deque(maxlen=100) for _ in range(self.n_agents)]}
        wt = 0
        # curr_ID_list = traci.vehicle.getIDList()
        for lane_id in self.lane_ids:
            wt += traci.lane.getWaitingTime(lane_id)
        self.curr_ID_list = traci.vehicle.getIDList()

        for id_idx, id in enumerate(self.CAVs_id_list):
            if id not in self.curr_ID_list:
                continue
            pos = list(traci.vehicle.getPosition(id))
            speed = traci.vehicle.getSpeed(id)
            self.obs_temp["pos"][id].append(pos)
            self.obs_temp["speed"][id].append(speed)
            dist_list = self._get_distance_list_between_ego_and_other_vehicles(id, exclude_same_phase=True)
            if min(dist_list) <= self.safe_dist:  # bit 3
                self.obs_temp["min_dist"][id].append(min(dist_list) / self.safe_dist)
            else:
                self.obs_temp["min_dist"][id].append(1.0)
            self.obs_temp["wt"][id].append(wt)
            self.obs_temp["enter_flag"][id].append(1.0 if self.enter_flag_dict[id] else 0.0)
            self.obs_temp["leave_flag"][id].append(1.0 if self.leave_flag_dict[id] else 0.0)

    def get_obs(self):
        """ Returns all agent observations in a list """
        observations = np.array(self._get_raw_observations(), dtype=np.float64)
        observations = self._normalize_observation_by_max(observations)
        return observations

    def _get_raw_observations(self):
        """
        each bit of observations:
        (x, y, v, safe_distance, waiting_time, enter_flag, leave_flag)

        if consider communication delay, then get observation from the following data structure:
        # self.obs_temp = {"pos": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "speed": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "min_dist": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "wt": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "enter_flag": [deque(maxlen=100) for _ in range(self.n_agents)],
        #                  "leave_flag": [deque(maxlen=100) for _ in range(self.n_agents)]}
        """
        # observations = [[0 for _ in range(self.obs_dim_per_veh)] for _ in range(self.total_agents)]
        observations = {str(t): [0 for _ in range(self.obs_dim_per_veh)] for t in range(self.total_agents)}
        wt = 0
        # curr_ID_list = traci.vehicle.getIDList()
        for id in self.lane_ids:
            wt += traci.lane.getWaitingTime(id)

        self.curr_ID_list = traci.vehicle.getIDList()
        for id_idx, id in enumerate(self.CAVs_id_list):
            # set the random route bit:
            observations[id][self.leave_flag_idx] = float(self.veh_intentions[id])
            # id_idx = self.CAVs_id_list.index(id)
            if id not in self.curr_ID_list:
                continue
            # print("step is: ",self.step_num, 'current ID list: ', self.curr_ID_list, "the position info: ", traci.vehicle.getPosition(id))
            # get (x,y,v) for each vehicle
            if not self.comm_lag:  # when there is no communication lag:
                if traci.vehicle.getPosition(id):  # if get the state information
                    observations[id][:self.pos_related_dim] = list(traci.vehicle.getPosition(id))  # bit 0,1
                    observations[id][self.vel_end_idx - 1] = traci.vehicle.getSpeed(id)  # bit 2
                # process safe distance
                dist_list = self._get_distance_list_between_ego_and_other_vehicles(id, exclude_same_phase=True)
                if not dist_list:  # bit 3
                    observations[id][self.safe_dist_idx - 1] = 1.0
                elif min(dist_list) <= self.safe_dist:
                    observations[id][self.safe_dist_idx - 1] = min(dist_list) / self.safe_dist
                else:
                    observations[id][self.safe_dist_idx - 1] = 1.0

                observations[id][self.wt_dim_idx - 1] = wt  # bit 4
                observations[id][self.enter_flag_idx - 1] = 1.0 if self.enter_flag_dict[id] else 0.0  # bit 5
                observations[id][self.leave_flag_idx - 1] = 1.0 if self.leave_flag_dict[id] else 0.0  # bit 6
            else:  # when there are communication lags
                # todo: use the delayed information as observation
                cur_length = len(self.obs_temp["pos"][id])
                observations[id][:self.pos_related_dim] = self.obs_temp["pos"][id][0] \
                    if len(self.obs_temp["pos"][id]) <= self.comm_lag else \
                    self.obs_temp["pos"][id][np.random.randint(-1 - int(abs(self.comm_lag)), 0)]  # bit 0,1
                observations[id][self.vel_end_idx - 1] = self.obs_temp["speed"][id][0] \
                    if len(self.obs_temp["speed"][id]) <= self.comm_lag \
                    else self.obs_temp["speed"][id][np.random.randint(-1 - int(abs(self.comm_lag)), 0)]  # bit 2
                observations[id][self.safe_dist_idx - 1] = self.obs_temp["min_dist"][id][0] \
                    if len(self.obs_temp["min_dist"][id]) <= self.comm_lag \
                    else self.obs_temp["min_dist"][id][np.random.randint(-1 - int(abs(self.comm_lag)), 0)]
                observations[id][self.wt_dim_idx - 1] = self.obs_temp["wt"][id][0] \
                    if len(self.obs_temp["wt"][id]) <= self.comm_lag \
                    else self.obs_temp["wt"][id][np.random.randint(-1 - int(abs(self.comm_lag)), 0)]  # bit 4
                observations[id][self.enter_flag_idx - 1] = self.obs_temp["enter_flag"][id][0] \
                    if len(self.obs_temp["enter_flag"][id]) <= self.comm_lag \
                    else self.obs_temp["enter_flag"][id][
                    np.random.randint(-1 - int(abs(self.comm_lag)), 0)]  # bit 5
                observations[id][self.leave_flag_idx - 1] = self.obs_temp["leave_flag"][id][0] \
                    if len(self.obs_temp["leave_flag"][id]) <= self.comm_lag \
                    else self.obs_temp["leave_flag"][id][
                    np.random.randint(-1 - int(abs(self.comm_lag)), 0)]  # bit 6
        raw_obs = []
        for id, obs in observations.items():
            raw_obs.append(obs)
        return raw_obs

    def _normalize_observation_by_max(self, raw_obs):  # raw_obs: [8, obs_dim]
        # obs_nmlz = []
        for idv_obs in raw_obs:  # raw_obs is numpy.array
            idv_obs[:self.pos_related_dim] /= self.max_lane_length
            idv_obs[self.vel_end_idx - 1] /= self.max_speed
            if idv_obs[self.wt_dim_idx - 1] >= self.waiting_steps_threshold:
                idv_obs[self.wt_dim_idx - 1] = 1.0
            else:
                idv_obs[self.wt_dim_idx - 1] /= self.waiting_steps_threshold
            # obs_cat = np.concatenate((pos, [vel], idv_obs[self.vel_end_idx:])).tolist()
            # obs_nmlz.append(obs_cat)
        # return raw_obs.tolist()
        return raw_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        obs_agent = self.get_obs()[agent_id]
        return obs_agent

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return int(self.observation_space.shape[0] / self.total_agents)

    def get_state(self):
        # observations = np.array(self.get_obs())
        # state = observations.reshape((1, self.n_agents * self.obs_dim_per_veh)) # [8,n]
        state = np.array(self.get_obs()).reshape((1, -1))  # [8,n]
        if state.size == 0:
            state = np.zeros((1, self.n_agents * self.obs_dim_per_veh))
        # state = np.concatenate((observations)).tolist()
        # print("state is: ", state)
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return int(self.observation_space.shape[0])

    def get_avail_actions(self):
        # currently assume all vehicles have the same available actions
        # [accelerate, keep constant speed, decelerate]
        avail_actions = []
        # =============== mask the actions by some rules, similar to the oracle information ============ #
        if len(self.CAVs_id_list) <= self.total_agents:
            for idx, id in enumerate(self.CAVs_id_list):
                avail_action = self.get_avail_agent_actions_general_signal(agent_id=id) if self.use_virtual_signal \
                    else self.get_avail_agent_actions_no_mask(agent_id=id)
                # print(id, avail_action)
                avail_actions.append(avail_action)
        # print(avail_actions)
        # for the rest of the non-controlled vehicles
        for i in range(self.total_agents):
            if str(i) in self.CAVs_id_list:
                continue
            else:
                avail_actions.append(np.array([1 for _ in range(self.action_dim_per_veh)], dtype=np.int32).tolist())
        return avail_actions  # [1 1 1]

    def get_avail_agent_actions_no_mask(self, agent_id):
        # agent_id: str
        # todo: agent_id never used
        """ Returns the available actions for agent_id """
        # avail_actions correspond to:
        # [smaller_accel, small_accel, big_accel, 0, -|smaller_accel|, -|small_accel|, -|big_accel|]
        # agent_id = str(agent_str_id)
        # agent_index = self.CAVs_id_list.index(agent_id)
        avail_actions = np.array([1 for _ in range(self.action_dim_per_veh)], dtype=np.int32)
        return avail_actions

    # def get_avail_agent_actions(self, agent_id):
    #     # agent_id: str
    #     # todo: agent_id never used
    #     """ Returns the available actions for agent_id """
    #     # avail_actions correspond to:
    #     # [smaller_accel, small_accel, big_accel, 0, -|smaller_accel|, -|small_accel|, -|big_accel|]
    #     # agent_id = str(agent_str_id)
    #     # agent_index = self.CAVs_id_list.index(agent_id)
    #     avail_actions = np.array([1 for _ in range(self.action_dim_per_veh)], dtype=np.int32)
    #     min_danger_dist = 1.5 * self.safe_dist
    #     may_danger_dist = 2.0 * self.safe_dist
    #     fast_pass_dist = 3.5 * self.safe_dist  # fast pass the intersection
    #     # currently assume all vehicles have the same available actions
    #     # higher-level decision:
    #     avail_actions_tmp = np.array([0 for _ in range(self.action_dim_per_veh)])
    #     # if agent_id not in self.curr_ID_list:
    #     #     return avail_actions
    #     if (self.leave_flag_dict[agent_id]):  # if agent leaves the intersection
    #         avail_actions_tmp[int(self.action_dim_per_veh / 2)] = 1  # keep the current speed
    #         avail_actions = avail_actions_tmp.copy()
    #         return avail_actions
    #     within_intersection = self.enter_flag_dict[agent_id] and (not self.leave_flag_dict[agent_id])
    #     not_enter_intersection = (not self.enter_flag_dict[agent_id]) and (not self.leave_flag_dict[agent_id])
    #     # Before the vehicle enters the junction zone:
    #     keep_speed_indx = int(self.action_dim_per_veh / 2)
    #     if not_enter_intersection:
    #         if agent_id in self.virtual_signal_veh_pair[0]:
    #             if 0 <= self.step_num < self.signal_cycle_num / self.time_step:
    #                 avail_actions_tmp[keep_speed_indx - 2: keep_speed_indx] = 1  # accelerate with bigger accelerations
    #                 avail_actions = avail_actions_tmp.copy()
    #             elif self.step_num >= 1 * (self.signal_cycle_num / self.time_step):
    #                 avail_actions_tmp[0] = 1  # drive with smaller accelerations
    #                 avail_actions_tmp[keep_speed_indx] = 1  # or keep the speed
    #                 avail_actions = avail_actions_tmp.copy()
    #         elif agent_id in self.virtual_signal_veh_pair[1]:
    #             if 1 * (self.signal_cycle_num / self.time_step) <=\
    #                     self.step_num < 2 * (self.signal_cycle_num / self.time_step):
    #                 avail_actions_tmp[keep_speed_indx - 2: keep_speed_indx] = 1  # accelerate with bigger accelerations
    #                 avail_actions = avail_actions_tmp.copy()
    #             elif self.step_num >= 2 * (self.signal_cycle_num / self.time_step):
    #                 # avail_actions_tmp[0] = 1  # drive with several smaller accelerations
    #                 avail_actions_tmp[keep_speed_indx] = 1  # or keep the speed
    #                 avail_actions = avail_actions_tmp.copy()
    #             elif self.step_num < 1 * (self.signal_cycle_num / self.time_step):
    #                 avail_actions_tmp[keep_speed_indx: 2 * keep_speed_indx] = 1  # keep the current speed or decelerate
    #                 avail_actions = avail_actions_tmp.copy()
    #         elif agent_id in self.virtual_signal_veh_pair[2]:
    #             if 2 * (self.signal_cycle_num / self.time_step) <=\
    #                     self.step_num < 3 * (self.signal_cycle_num / self.time_step):
    #                 avail_actions_tmp[keep_speed_indx - 2: keep_speed_indx] = 1  # accelerate with bigger accelerations
    #                 avail_actions = avail_actions_tmp.copy()
    #             elif self.step_num >= 3 * (self.signal_cycle_num / self.time_step):
    #                 # avail_actions_tmp[0] = 1  # drive with several smaller accelerations
    #                 avail_actions_tmp[keep_speed_indx] = 1  # or keep the speed
    #                 avail_actions = avail_actions_tmp.copy()
    #             elif self.step_num < 2 * (self.signal_cycle_num / self.time_step):
    #                 avail_actions_tmp[keep_speed_indx: 2 * keep_speed_indx] = 1  # keep the current speed or decelerate
    #                 avail_actions = avail_actions_tmp.copy()
    #         elif agent_id in self.virtual_signal_veh_pair[3]:
    #             if 3 * (self.signal_cycle_num / self.time_step) <=\
    #                     self.step_num < 4 * (self.signal_cycle_num / self.time_step):
    #                 avail_actions_tmp[keep_speed_indx - 1] = 1  # accelerate with bigger accelerations
    #                 avail_actions = avail_actions_tmp.copy()
    #             elif self.step_num >= 4 * (self.signal_cycle_num / self.time_step):
    #                 # avail_actions_tmp[0] = 1  # drive with several smaller accelerations
    #                 avail_actions_tmp[keep_speed_indx] = 1  # or keep the speed
    #                 avail_actions = avail_actions_tmp.copy()
    #             elif self.step_num < 3 * (self.signal_cycle_num / self.time_step):
    #                 avail_actions_tmp[keep_speed_indx: 2 * keep_speed_indx] = 1  # keep the current speed or with decent decelerations
    #                 avail_actions = avail_actions_tmp.copy()
    #         return avail_actions
    #
    #     # After the vehicle enters the junction zone:
    #     # 1. wait when CAV drives into the danger zone (decelerate with max deceleration);
    #     if within_intersection:
    #         if self._within_danger_zone_at_intersection(vehID=agent_id, safe_distance=min_danger_dist) and \
    #                 traci.vehicle.getSpeed(agent_id) > 0.1 * self.max_speed:
    #             # if vehicles are in the dangerous zone, it must decelerate with max deceleration
    #             avail_actions_tmp[-1] = 1
    #             avail_actions = avail_actions_tmp.copy()
    #
    #         # 2. if intersection is crowded, CAV drives slowly;
    #         # elif self._within_danger_zone_at_intersection(vehID=agent_id, safe_distance=may_danger_dist) and \
    #         #         0.2 * self.max_speed > traci.vehicle.getSpeed(agent_id) >= 0.1 * self.max_speed:
    #         #     avail_actions_tmp[-int(self.action_dim_per_veh / 2)] = 1  # keep the speed
    #         #     avail_actions = avail_actions_tmp.copy()
    #         # elif self._within_danger_zone_at_intersection(vehID=agent_id, safe_distance=may_danger_dist) and \
    #         #         0.5 * self.max_speed > traci.vehicle.getSpeed(agent_id) >= 0.2 * self.max_speed:
    #         #     avail_actions_tmp[-int(self.action_dim_per_veh / 2) + 1] = 1  # decelerate with medium deceleration
    #         #     avail_actions = avail_actions_tmp.copy()
    #         # 3. if safe, pass the junction as fast as it can.
    #         elif self._within_danger_zone_at_intersection(vehID=agent_id, safe_distance=fast_pass_dist) and \
    #                 traci.vehicle.getSpeed(agent_id) < 0.5 * self.max_speed:
    #             avail_actions_tmp[int(self.action_dim_per_veh / 2) - 1] = 1  # drive fastest
    #             avail_actions = avail_actions_tmp.copy()
    #         else:
    #             avail_actions = [1 for _ in range(2)] + [0 for _ in range(self.action_dim_per_veh - 2)]
    #         # print(agent_id, avail_actions)
    #         return avail_actions
    #     return avail_actions
    #     # todo: when the vehicles enter the intersection, we extend their action spaces

    def get_avail_agent_actions_general_signal(self, agent_id):
        # agent_id: str
        # todo: agent_id never used
        """ Returns the available actions for agent_id """
        # avail_actions correspond to:
        # [smaller_accel, small_accel, big_accel, 0, -|smaller_accel|, -|small_accel|, -|big_accel|]
        # agent_id = str(agent_str_id)
        # agent_index = self.CAVs_id_list.index(agent_id)
        avail_actions = np.array([1 for _ in range(self.action_dim_per_veh)], dtype=np.int32)
        min_danger_dist = 1.5 * self.safe_dist
        may_danger_dist = 2.0 * self.safe_dist
        fast_pass_dist = 3.5 * self.safe_dist  # fast pass the intersection
        # currently assume all vehicles have the same available actions
        # higher-level decision:
        avail_actions_tmp = np.array([0 for _ in range(self.action_dim_per_veh)])
        # if agent_id not in self.curr_ID_list:
        #     return avail_actions
        if (self.leave_flag_dict[agent_id]):  # if agent leaves the intersection
            avail_actions_tmp[int(self.action_dim_per_veh / 2)] = 1  # keep the current speed
            avail_actions = avail_actions_tmp.copy()
            return avail_actions
        within_intersection = self.enter_flag_dict[agent_id] and (not self.leave_flag_dict[agent_id])
        not_enter_intersection = (not self.enter_flag_dict[agent_id]) and (not self.leave_flag_dict[agent_id])
        # Before the vehicle enters the junction zone:
        keep_speed_indx = int(self.action_dim_per_veh / 2)
        route = self.selected_routes_of_vehicles[agent_id][0]
        if not_enter_intersection:
            # if self._within_danger_zone_before_intersection(vehID=agent_id, safe_distance=min_danger_dist) and \
            #         traci.vehicle.getSpeed(agent_id) > 0.1 * self.max_speed:
            #     avail_actions_tmp[keep_speed_indx + 1:] = 1
            #     avail_actions = avail_actions_tmp.copy()
            #     return avail_actions

            if route in self.virtual_signal_veh_pair_general[0]:
                if 0 <= self.step_num < self.signal_cycle_num / self.time_step:
                    avail_actions_tmp[keep_speed_indx - 2: keep_speed_indx] = 1  # accelerate with bigger accelerations
                    avail_actions = avail_actions_tmp.copy()
                elif self.step_num >= 1 * (self.signal_cycle_num / self.time_step):
                    avail_actions_tmp[0] = 1  # drive with smaller accelerations
                    avail_actions_tmp[keep_speed_indx] = 1  # or keep the speed
                    avail_actions = avail_actions_tmp.copy()
            elif route in self.virtual_signal_veh_pair_general[1]:
                if 1 * (self.signal_cycle_num / self.time_step) <= \
                        self.step_num < 2 * (self.signal_cycle_num / self.time_step):
                    avail_actions_tmp[keep_speed_indx - 2: keep_speed_indx] = 1  # accelerate with bigger accelerations
                    avail_actions = avail_actions_tmp.copy()
                elif self.step_num >= 2 * (self.signal_cycle_num / self.time_step):
                    # avail_actions_tmp[0] = 1  # drive with several smaller accelerations
                    avail_actions_tmp[keep_speed_indx] = 1  # or keep the speed
                    avail_actions = avail_actions_tmp.copy()
                elif self.step_num < 1 * (self.signal_cycle_num / self.time_step):
                    avail_actions_tmp[keep_speed_indx: 2 * keep_speed_indx] = 1  # keep the current speed or decelerate
                    avail_actions = avail_actions_tmp.copy()
            elif route in self.virtual_signal_veh_pair_general[2]:
                if 2 * (self.signal_cycle_num / self.time_step) <= \
                        self.step_num < 3 * (self.signal_cycle_num / self.time_step):
                    avail_actions_tmp[keep_speed_indx - 2: keep_speed_indx] = 1  # accelerate with bigger accelerations
                    avail_actions = avail_actions_tmp.copy()
                elif self.step_num >= 3 * (self.signal_cycle_num / self.time_step):
                    # avail_actions_tmp[0] = 1  # drive with several smaller accelerations
                    avail_actions_tmp[keep_speed_indx] = 1  # or keep the speed
                    avail_actions = avail_actions_tmp.copy()
                elif self.step_num < 2 * (self.signal_cycle_num / self.time_step):
                    avail_actions_tmp[keep_speed_indx: 2 * keep_speed_indx] = 1  # keep the current speed or decelerate
                    avail_actions = avail_actions_tmp.copy()
            elif route in self.virtual_signal_veh_pair_general[3]:
                if 3 * (self.signal_cycle_num / self.time_step) <= \
                        self.step_num < 4 * (self.signal_cycle_num / self.time_step):
                    avail_actions_tmp[keep_speed_indx - 1] = 1  # accelerate with bigger accelerations
                    avail_actions = avail_actions_tmp.copy()
                elif self.step_num >= 4 * (self.signal_cycle_num / self.time_step):
                    # avail_actions_tmp[0] = 1  # drive with several smaller accelerations
                    avail_actions_tmp[keep_speed_indx] = 1  # or keep the speed
                    avail_actions = avail_actions_tmp.copy()
                elif self.step_num < 3 * (self.signal_cycle_num / self.time_step):
                    avail_actions_tmp[
                    keep_speed_indx: 2 * keep_speed_indx] = 1  # keep the current speed or with decent decelerations
                    avail_actions = avail_actions_tmp.copy()
            return avail_actions

        # After the vehicle enters the junction zone:
        # 1. wait when CAV drives into the danger zone (decelerate with max deceleration);
        if within_intersection:
            if self._within_danger_zone_at_intersection(vehID=agent_id, safe_distance=min_danger_dist) and \
                    traci.vehicle.getSpeed(agent_id) > 0.1 * self.max_speed:
                # if vehicles are in the dangerous zone, it must decelerate with max deceleration
                avail_actions_tmp[-1] = 1
                avail_actions = avail_actions_tmp.copy()

            # 2. if intersection is crowded, CAV drives slowly;
            # elif self._within_danger_zone_at_intersection(vehID=agent_id, safe_distance=may_danger_dist) and \
            #         0.2 * self.max_speed > traci.vehicle.getSpeed(agent_id) >= 0.1 * self.max_speed:
            #     avail_actions_tmp[-int(self.action_dim_per_veh / 2)] = 1  # keep the speed
            #     avail_actions = avail_actions_tmp.copy()
            # elif self._within_danger_zone_at_intersection(vehID=agent_id, safe_distance=may_danger_dist) and \
            #         0.5 * self.max_speed > traci.vehicle.getSpeed(agent_id) >= 0.2 * self.max_speed:
            #     avail_actions_tmp[-int(self.action_dim_per_veh / 2) + 1] = 1  # decelerate with medium deceleration
            #     avail_actions = avail_actions_tmp.copy()
            # 3. if safe, pass the junction as fast as it can.
            elif self._within_danger_zone_at_intersection(vehID=agent_id, safe_distance=fast_pass_dist) and \
                    traci.vehicle.getSpeed(agent_id) < 0.5 * self.max_speed:
                avail_actions_tmp[int(self.action_dim_per_veh / 2) - 1] = 1  # drive fastest
                avail_actions = avail_actions_tmp.copy()
            else:
                avail_actions = [1 for _ in range(2)] + [0 for _ in range(self.action_dim_per_veh - 2)]
            # print(agent_id, avail_actions)
            return avail_actions
        return avail_actions
        # todo: when the vehicles enter the intersection, we extend their action spaces

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.action_dim_per_veh if self.discrete else 1

    def reset(self, env_args=None):
        """ Returns initial observations and states"""
        print(self.intention_probs)
        if self.episode_num == 0:
            if self.run_curricula:
                self.random_behavior = env_args["random_behavior"]
                self.comm_lag = env_args["comm_lag"]
                self.n_agents = env_args["n_agents"]
                self.CAVs_id_list = env_args["CAVs_id_list"]
                self.intention_probs = env_args["intention_probs"]
            # if self.phase is not None:
            #     self.comm_lag = self.comm_lag_curricula[self.phase]
            self.first_start()
            self.episode_num += 1
        else:
            self.done = False
            self.step_num = 0
            self.episode_num += 1
            # self.control_len = 1/2 * self.lane_length
            # self.enter_flag_list_tmp = [False for _ in range(self.n_agents)]
            # self.leave_flag_list_tmp = [False for _ in range(self.n_agents)]
            # self.enter_flag_list = [False for _ in range(self.n_agents)]
            # self.leave_flag_list = [False for _ in range(self.n_agents)]
            self.veh_intentions = OrderedDict({})
            self.env_infos = {"complete_flag": float(False), "collisions": 0.0}
            self.enter_flag_dict_tmp = {str(t): False for t in range(self.total_agents)}
            self.leave_flag_dict_tmp = {str(t): False for t in range(self.total_agents)}
            self.enter_flag_dict = {str(t): False for t in range(self.total_agents)}
            self.leave_flag_dict = {str(t): False for t in range(self.total_agents)}
            self.selected_routes_of_vehicles = OrderedDict({vehID: [] for vehID in self.CAVs_id_list})
            self.complete_flag = False
            traci.load(self.reloadCmd)
            self.shape_intersection = traci.junction.getShape('J1')
            self.len_intersection = 2 * max([max(ele) for ele in self.shape_intersection])
            self.lane_length = [traci.lane.getLength(id) for id in self.lane_ids][
                0]  # assume all lanes share the same length
            if self.run_curricula:
                self.random_behavior = env_args["random_behavior"]
                self.comm_lag = env_args["comm_lag"]
                self.n_agents = env_args["n_agents"]
                self.CAVs_id_list = env_args["CAVs_id_list"]
            # if self.phase is not None:
            #     self.comm_lag = self.comm_lag_curricula[self.phase]

            self._spawn_CAVs_in_sumo(random_behavior=self.random_behavior)
            traci.simulationStep()  # avoid all zero observations
            self.obs_temp = {"pos": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                             "speed": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                             "min_dist": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                             "wt": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                             "enter_flag": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                             "leave_flag": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                             }
            self._update_info()
            self.curr_ID_list = traci.vehicle.getIDList()
            # obs = self.get_obs()
        # return obs

    def _set_enter_and_leave_flags(self):
        # assign true flag to enter_flag_list from tmp variable
        for id_idx, id in enumerate(self.CAVs_id_list):
            if id not in self.curr_ID_list:  # if one CAV goes out of scope, continue.
                continue
            self._if_veh_enters_the_junction(vehID=id)  # set tmp variable
            # self.enter_flag_list = self.enter_flag_list_tmp.copy()
            if self.enter_flag_dict_tmp[id]:
                self.enter_flag_dict[id] = True
            # ------------------------------------------ #
            if self.enter_flag_dict[id]:  # the vehicle has entered the intersection
                self._if_veh_leaves_the_junction(id)
                if self.leave_flag_dict_tmp[id]:
                    self.leave_flag_dict[id] = True

    def first_start(self):
        # self.sumoCmd[0] = 'sumo-gui'
        traci.start(self.sumoCmd)
        self.shape_intersection = traci.junction.getShape('J1')
        self.len_intersection = 2 * max([max(ele) for ele in self.shape_intersection])
        self.lane_length = [traci.lane.getLength(id) for id in self.lane_ids][0]  # assume all lanes share the same length
        # add CAVs
        self._spawn_CAVs_in_sumo(random_behavior=self.random_behavior)
        traci.simulationStep()  # avoid all zero observations
        self.obs_temp = {"pos": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "speed": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "min_dist": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "wt": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "enter_flag": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         "leave_flag": {str(t): deque(maxlen=100) for t in range(self.total_agents)},
                         }
        self.env_infos = {"complete_flag": float(False), "collisions": 0.0}
        self.step_num = 0
        self.lane_length = [traci.lane.getLength(id) for id in self.lane_ids][
            0]  # assume all lanes share the same length
        self.lane_width = [traci.lane.getWidth(id) for id in self.lane_ids][0]  # assume all lanes share the same length
        self.control_len = 1 / 2 * self.lane_length
        self.curr_ID_list = traci.vehicle.getIDList()
        self.complete_flag = False
        self._update_info()
        # return self.get_obs()

    def _spawn_CAVs_in_sumo(self, random_behavior=False):
        if self.n_agents <= self.total_agents:
            for i in self.CAVs_id_list:  # i is of string type
                self._add_vehicle_in_sumo(i, if_cav=True, random_behavior=random_behavior)
                # traci.simulationStep()
        # for the rest of the vehicles:
        for i in range(self.total_agents):
            if str(i) in self.CAVs_id_list:
                continue
            else:
                self._add_vehicle_in_sumo(str(i), if_cav=False, random_behavior=random_behavior)

    def _add_vehicle_in_sumo(self, vehID, if_cav=True, random_behavior=False):
        intention = np.random.choice([i for i in range(self.intention_dim)], p=self.intention_probs)
        self.veh_intentions[vehID] = intention
        routes = self.routes_of_vehicles[vehID][intention] if random_behavior else self.routes_fixed[vehID]
        self.selected_routes_of_vehicles[vehID].append(routes)
        traci.vehicle.add(vehID, routes, typeID=self.type_name,
                          depart=self.vehicle_spawn_infos["depart"],
                          departPos=self.vehicle_spawn_infos["departPos"],
                          departLane=self.CAVs_departLane[vehID][-1],
                          departSpeed=self.vehicle_spawn_infos["departSpeed"])
        if if_cav:
            traci.vehicle.setColor(vehID, ['255', '0', '0'])

    def _get_rewards(self, next_observations):
        rewards = [0 for _ in range(self.n_agents)]
        for idx, id in enumerate(self.CAVs_id_list):
            if id in self.curr_ID_list:
                if next_observations[idx][-1] < 0.1:
                    rewards[idx] -= 5
                rewards[idx] += next_observations[idx][-1]
                if self._within_danger_zone_at_intersection(index=idx, vehID=id):
                    rewards[idx] -= 10
        return rewards

    def _derive_accel_from_chosen_actions_idxes(self, vehID, actions_idxes):
        # vehID : int
        # max_speed = traci.vehicle.getMaxSpeed(vehID)
        action_idx = np.array(actions_idxes[int(vehID)])  # one-hot vector
        # effective_action_idx = np.argwhere(action==1).reshape(-1)
        if (action_idx >= 0) and (action_idx <= int(((self.action_dim_per_veh - 1) / 2 - 1))):  # accel
            accel = self.accel_res[int(action_idx)]
        elif action_idx == (self.action_dim_per_veh - 1) / 2:  # keep current speed
            accel = 0
        elif (action_idx >= (self.action_dim_per_veh - 1) / 2 + 1) and (
                action_idx <= self.action_dim_per_veh - 1):  # decel
            accel = -self.accel_res[int(action_idx) - self.action_dim_per_veh]
        # print('accel is: ',accel)
        return accel

    def _get_shared_reward(self, next_observations):
        reward = 0
        # wt_list = []
        wt = 0
        self.v_threshold = 2 / self.max_speed
        if_danger_dict = {t: False for t in self.CAVs_id_list}

        for idx, id in enumerate(self.CAVs_id_list):
            if id in self.curr_ID_list:
                # ---------- quickly through the intersection -------------#
                if next_observations[idx][self.vel_end_idx - 1] < self.v_threshold:
                    reward -= 0.5
                # ====== reward acceleration =======#
                # else:
                #     reward += next_observations[idx][self.vel_end_idx - 1] - self.v_threshold
                # --------------- collision test given the safe distance -------------#
                danger = self._within_danger_zone_at_intersection(vehID=id, safe_distance=self.safe_dist)
                if_danger_dict[id] = danger
                if danger:
                    reward -= 5
        # --------------------- penalize long waiting time -------------------#
        for id in self.lane_ids:
            wt += traci.lane.getWaitingTime(id)
        # wt_list.append(wt)
        reward -= 0.05 * wt
        # --------------------- Need less time to achieve the task -------------------#
        reward -= 0.005
        # print(self.step_num, wt)
        # --------------------- if task is successfully completed ------------------#
        """Explanation of this case: all vehicles leave the intersection without near collisions"""
        enter_flags, leave_flags = [], []
        for id in self.CAVs_id_list:
            enter_flags.append(self.enter_flag_dict[id])
            leave_flags.append(self.leave_flag_dict[id])
        if all(enter_flags) and all(leave_flags) and \
                traci.simulation.getCollidingVehiclesNumber() == 0:
            reward += (self.episode_limit - self.step_num) + 20
            # reward += len(self.CAVs_id_list) * 10
            self.complete_flag = True
            self.env_infos["complete_flag"] = float(self.complete_flag)
        # elif True not in if_danger_list and all(self.leave_flag_list) and self.step_num <= int(self.episode_limit):
        #     reward += 100
        # ---------------- Encourage more vehicles pass through the intersection ------------------#
        # vehicles_through_step = 0
        # for i, flag in enumerate(self.leave_flag_list):
        #     if flag == True and not if_danger_list[i]:
        #         vehicles_through_step += 1
        # vehicles_through = vehicles_through_step - self.vehicles_through
        # self.vehicles_through = vehicles_through_step
        # # print(vehicles_through)
        # reward += vehicles_through
        # ------------ Penalize the case when there are vehicles not passing through the intersection ------------#
        vehicles_through_step = 0
        # for id, flag in self.leave_flag_dict.items():
        for id in self.CAVs_id_list:
            flag = self.leave_flag_dict[id]
            if flag is True and not if_danger_dict[id]:
                vehicles_through_step += 1
        vehicles_through = vehicles_through_step - self.vehicles_through
        self.vehicles_through = vehicles_through_step
        # print(vehicles_through)
        # veh_not_through = self.n_agents - vehicles_through_step
        # reward -= 0.01 * veh_not_through
        reward += 0.1 * vehicles_through
        return reward

    def _get_done(self):
        # ------------- basic decision of termination of an episode -------------#
        done = self.done
        if done:  # if the initial done condition is true
            return done
        if () in self.curr_ID_list and self.step_num > 4:
            done = True
        # ------------ decide if the waiting time is too long --------------#
        wt = 0
        for id in self.lane_ids:
            wt += traci.lane.getWaitingTime(id)
        if wt > self.waiting_steps_threshold:
            done = True
        # ------------- vehicles do not collide with each other at intersection --------#
        # for id in self.CAVs_id_list:
        #     if id not in self.curr_ID_list:
        #         continue
        #     if self._within_danger_zone_at_intersection(id, safe_distance=self.safe_dist):
        #         done = True
        if traci.simulation.getCollidingVehiclesNumber():
            done = True
            self.collision_times = traci.simulation.getCollidingVehiclesNumber()
            self.env_infos["collisions"] = self.collision_times
        # -------------- decide if all cars leave the intersection --------#
        leave_flags = []
        for id in self.CAVs_id_list:
            leave_flags.append(self.leave_flag_dict[id])
        if all(leave_flags):
            done = True
        # ----------- decide if step_num is beyond the scope  -----------#
        if self.step_num >= self.episode_limit:
            done = True
        return done

    # def _is_success(self):
    #     if all(self.leave_flag_list) and traci.simulation.getCollidingVehiclesNumber() == 0:
    #         return True
    def _get_distance_list_between_ego_and_other_vehicles(self, vehID, exclude_same_phase=False):
        dist_list = []
        pos0 = np.array(traci.vehicle.getPosition(vehID))
        # for the case including all the vehicles
        if not exclude_same_phase:
            for id in self.curr_ID_list:
                if id != vehID:
                    pos1 = np.array(traci.vehicle.getPosition(id))
                    dist = np.linalg.norm(pos0 - pos1)
                    dist_list.append(dist)
            return dist_list

        # for the case EXCLUDING all the vehicles
        ego_route = traci.vehicle.getRouteID(vehID)
        phase_id = None
        for (phase_num, routes) in self.virtual_signal_veh_pair_general.items():
            if ego_route in routes:
                phase_id = phase_num
                break
        cur_ID_list = list(self.curr_ID_list).copy()
        # delete the unrelated vehicles
        for id in self.curr_ID_list:
            cur_route = traci.vehicle.getRouteID(id)
            if cur_route in self.virtual_signal_veh_pair_general[phase_id]:
                cur_ID_list.remove(id)

        for id in cur_ID_list:
            if id != vehID:
                pos1 = np.array(traci.vehicle.getPosition(id))
                dist = np.linalg.norm(pos0 - pos1)
                dist_list.append(dist)
        return dist_list

    # def _get_distance_list_between_ego_and_other_vehicles(self, vehID, exclude_same_phase=False):
    #     dist_list = []
    #     pos0 = np.array(traci.vehicle.getPosition(vehID))
    #     vehID_in_same_phase = self.same_directions[vehID]
    #     for id in self.curr_ID_list:
    #         if exclude_same_phase:
    #             if vehID_in_same_phase == id:
    #                 continue
    #         if id != vehID:
    #             pos1 = np.array(traci.vehicle.getPosition(id))
    #             dist = np.linalg.norm(pos0 - pos1)
    #             dist_list.append(dist)
    #     return dist_list

    def _get_ego_dist_list_vehicles_exclude_latters_before_intersection(self, vehID, exclude_same_direction=False):
        dist_list = []
        pos0 = np.array(traci.vehicle.getPosition(vehID))
        vehIDs_in_same_lane = list(traci.lane.getLastStepVehicleIDs(self.CAVs_departLane[vehID])) # include vehID
        vehID_same_direction = self.same_directions[vehID] # vehicles in differen lane but the same direction
        vehIDs_same_road_diff_lane = list(traci.lane.getLastStepVehicleIDs(self.CAVs_departLane[vehID_same_direction]))
        # split the list
        if vehID in vehIDs_in_same_lane:
            split_index = vehIDs_in_same_lane.index(vehID)
            excluded_ids = vehIDs_in_same_lane[:split_index]
            excluded_ids.extend(vehIDs_same_road_diff_lane)
            for id in self.curr_ID_list:
                if id != vehID:
                    if exclude_same_direction:
                        if id in excluded_ids:
                            continue
                    pos1 = np.array(traci.vehicle.getPosition(id))
                    dist = np.linalg.norm(pos0 - pos1)
                    dist_list.append(dist)
        return dist_list

    def _within_danger_zone_at_intersection(self, vehID, safe_distance=3.0):
        # vehID: str
        # distance_matrix = self._get_distance_matrix_between_any_2_vehicles_with_repeat()
        within_intersection = self.enter_flag_dict[vehID] and (not self.leave_flag_dict[vehID])
        # logging.info('If all vehicles are within the intersection: '+ str(within_intersection))
        dist_list = self._get_distance_list_between_ego_and_other_vehicles(vehID, exclude_same_phase=True)
        # print(dist_list)
        if not dist_list:  # if only one car is left, then it has no danger
            within_intersection = False
        within_danger_zone = False
        if within_intersection:
            dist = min(dist_list)
            # print(vehID, dist)
            if dist <= safe_distance:
                within_danger_zone = True
        return within_danger_zone

    def _within_danger_zone_before_intersection(self, vehID, safe_distance=3.0):
        # vehID: str
        # distance_matrix = self._get_distance_matrix_between_any_2_vehicles_with_repeat()
        # logging.info('If all vehicles are within the intersection: '+ str(within_intersection))
        # not_in_intersection = not self.enter_flag_dict[vehID] and (not self.leave_flag_dict[vehID])
        # dist_list = []
        # if not_in_intersection:
        dist_list = self._get_ego_dist_list_vehicles_exclude_latters_before_intersection(vehID,
                                                                                         exclude_same_direction=True)
        # print(dist_list)
        if not dist_list:  # if only one car is left, then it has no danger
            return False
        within_danger_zone = False
        dist = min(dist_list)
        # print(vehID, ": ", dist)
        if dist <= safe_distance:
            within_danger_zone = True
        return within_danger_zone

    def _if_veh_enters_the_junction(self, vehID):
        # vehID: str
        # veh_index = self.CAVs_id_list.index(vehID)
        curr_lane_id = traci.vehicle.getLaneID(vehID)
        if self.junction_name in curr_lane_id:
            self.enter_flag_dict_tmp[vehID] = True

    def _if_veh_leaves_the_junction(self, vehID):  # must be used after vehicle enters the intersection zone
        # veh_index = self.CAVs_id_list.index(vehID)
        curr_lane_id = traci.vehicle.getLaneID(vehID)
        if self.junction_name not in curr_lane_id:
            self.leave_flag_dict_tmp[vehID] = True

    def render(self):
        raise NotImplementedError

    def close(self):
        traci.close(wait=False)
        self.episode_num = 0
        # raise NotImplementedError

    # def seed(self, seed=None):
    #     self.np_random, self.seed = gym.utils.seeding.np_random(self.seed)
    #     return [seed]

    def save_replay(self):
        raise NotImplementedError
