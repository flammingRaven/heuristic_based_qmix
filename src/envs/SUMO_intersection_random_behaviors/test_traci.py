# this file is used to test the vehicle-related api of traci
import traci
import os,sys
import time
from sumolib import checkBinary
import numpy as np
import traci.constants as tc

# cfg_dir = "G:\\Reinforcement learning practice with Tensorflow\\SUMO related\\sumo_toturial_file-master\\sumo_toturial_file-master\\25_traci_usage\\exa.sumocfg"
cfg_dir = "/home/ghz/PycharmProjects/sumo_carla_cosim_marl_curriculum/src/envs/SUMO_intersection_random_behaviors/main.sumocfg"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools") # get the path of tools containing traci
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable SUMO_HOME")

show_gui = True

if not show_gui:
    sumoBinary = checkBinary("sumo")
else:
    sumoBinary = checkBinary("sumo-gui")

traci.start([sumoBinary, "-c", cfg_dir, '--collision.check-junctions'])
# traci.start([sumoBinary, "-c", cfg_dir])
accel_list = np.linspace(3,20,8)
lane_ids = ["L0_0","L0_1","L1_0","L1_1","L2_0","L2_1","L3_0","L3_1",
           "L4_0","L4_1","L5_0","L5_1","L6_0","L6_1","L7_0","L7_1",]
routes = ["route_WE", "route_WN", "route_EW", "route_ES",
          "route_NE", "route_NS", "route_SW", "route_SN"]
# routes_start_lanes = {"route_WE":{"L6_0":,"L6_1"},
#                       "route_WN":[], "route_EW", "route_ES",
#           "route_NE", "route_NS", "route_SW", "route_SN"}
n_vehicles = 8
speed_list = [2 for _ in range(n_vehicles)]
CAVs_id_list = [str(i) for i in range(n_vehicles)]
type_name = "vehicle.tesla.model3"
for i in range(n_vehicles):
    traci.vehicle.add(str(i), routes[i], typeID=type_name, depart="0", departPos="30.0", departLane="best")
# traci.simulationStep()

for step in range(300):
    time.sleep(0.1)
    # print(traci.simulation.getMinExpectedNumber())
    traci.simulationStep(step * 0.2)
    simulation_curr_time = traci.simulation.getTime()
    # print("The current time is: ", simulation_curr_time)
    #
    # spawned_actors = traci.simulation.getDepartedIDList()
    # print(spawned_actors)
    all_vehicle_id = traci.vehicle.getIDList()
    # print(all_vehicle_id)
    for id in CAVs_id_list:
        # curr_lane_id = traci.vehicle.getLaneID(id)
        if id not in all_vehicle_id:
            continue
        traci.vehicle.setSpeedMode(id, 32)
        traci.vehicle.setLaneChangeMode(id, 256)
        # traci.vehicle.setSpeed(vehID=id, speed=5)
    # print("Collision information: ", traci.simulation.getCollisions())
    # print("Collision information: ", traci.simulation.getCollidingVehiclesNumber())
    # print([traci.lane.getLastStepHaltingNumber(id) for id in lane_ids])
    # print(traci.vehicle.getRouteID('0'))
    # print(traci.lane.getLastStepVehicleIDs(lane_ids[0]))
    # print(list(traci.lane.getLastStepVehicleIDs(lane_ids[0])))
    # print([traci.lane.getLastStepVehicleIDs(id) for id in lane_ids])
    # print("Fuel consumption: ", traci.vehicle.getFuelConsumption('0'))
    print("Acceleration: ", traci.vehicle.getAcceleration('0'))
    collidingVehIDs = traci.simulation.getCollidingVehiclesIDList()
    if collidingVehIDs:
        print("Collision IDlist: ", collidingVehIDs)
    #     print(id, curr_lane_id)
    # num_veh = len(all_vehicle_id)
    # print('The current time is: ',simulation_curr_time, "The step is: ", step, "The Vehicle ID list is: ", all_vehicle_id)
    # all_vehicle_route = [(i, traci.vehicle.getRouteID(i)) for i in all_vehicle_id]
    # print(all_vehicle_route)
    #
    # for idx, i in enumerate(all_vehicle_id):
    #      # set speed mode and let RL fully takes over the control task
    #     traci.vehicle.setSpeedMode(i, 32) # all checks off -> [1 0 0 0 0 0] -> Speed Mode = 32
    #     # disable all autonomous changing but still handle safety checks, 512 for collision avoidance and safety-gap enforcement
    #     traci.vehicle.setLaneChangeMode(i, 256) # 1024
    # #
    #     # traci.vehicle.setAccel(vehID=i, accel=accel_list[idx]) # set maximum acceleration
    # for i, speed in enumerate(speed_list):
    #     # all_vehicle_id = traci.vehicle.getIDList()
    #     if str(i) in all_vehicle_id:
    #         traci.vehicle.setSpeed(vehID=str(i), speed=15)

    step += 1
    #
    # # for i in all_vehicle_id[4:]:
    # #     traci.vehicle.setSpeed(vehID=i, speed=11)
    # # print("The accelerations of all vehicles are: ", [traci.vehicle.getAcceleration(i) for i in all_vehicle_id])
    # print("The Speed of all vehicles are: ", [traci.vehicle.getSpeed(i) for i in all_vehicle_id])
    # junctionID = traci.junction.getIDList()

    # shape_intersection = traci.junction.getShape('J1')
    # len_intersection = max([max(ele) for ele in shape_intersection])
    # print('len_intersection: ', len_intersection)

    # print("The lane id of vehicles: ", [traci.vehicle.getLaneID(i) for i in all_vehicle_id])
    # # print("The distance each vehicle has traveled: ", [traci.vehicle.getDistance(i) for i in all_vehicle_id])
    # # dist_list= []
    # #
    # # for i in range(num_veh - 1):
    # #     extra_loop_nums = num_veh - i - 1
    # #     for k in range(extra_loop_nums):
    # #         pos0 = np.array(traci.vehicle.getPosition(all_vehicle_id[i]))
    # #         pos1 = np.array(traci.vehicle.getPosition(all_vehicle_id[k+1+i]))
    # #         # index += 1
    # #         dist = np.linalg.norm(pos0 - pos1)
    # #         dist_list.append(dist)
    #
    # dist_matrix = [[] for _ in range(num_veh)]
    # for i in range(num_veh):
    #     pos0 = np.array(traci.vehicle.getPosition(all_vehicle_id[i]))
    #     for j in range(num_veh):
    #         if j != i:
    #             pos1 = np.array(traci.vehicle.getPosition(all_vehicle_id[j]))
    #             dist = np.linalg.norm(pos0 - pos1)
    #             dist_matrix[i].append(dist)
    #
    # print("The distance between 2 vehicles: ", dist_matrix)


    # print('The width of lane: ', [traci.lane.getWidth(id) for id in lane_ids])
    # print("The action step length of 1-st vehicle: ", traci.vehicle.getActionStepLength('0'))
    # print("The speed mode and lane change mode of 1-th vehicle: ", traci.vehicle.getSpeedMode('0'), traci.vehicle.getLaneChangeMode('0'))

    # traci.vehicle.setLaneChangeMode()
    # all_vehicle_position = [(i, traci.vehicle.getPosition(i)) for i in all_vehicle_id]
    # print("The position of all vehicles are: ", all_vehicle_position)
    # waiting_time = [traci.vehicle.getAccumulatedWaitingTime(i) for i in all_vehicle_id]
    # print("step: ", step, " The waiting time is: ", waiting_time)
    # print("Is vehicles stopped? ",[traci.vehicle.isStopped(i) for i in all_vehicle_id])
    # print("Collision information: ", traci.simulation.getCollidingVehiclesNumber())
    # print([traci.vehicle.getRouteID(i) for i in all_vehicle_id])
    # print("Waiting time is: ", [traci.lane.getWaitingTime(id) for id in lane_ids])
    # lane_id_list = traci.lane.getIDList()
    # lane_length_list = [traci.lane.getLength(id) for id in lane_ids]
    # print("lane_ids: ", lane_id_list, len(lane_id_list))
    # print("lane length: ", lane_length_list, len(lane_length_list))
    # print("Colliding vehicles: ", traci.simulation.getCollidingVehiclesNumber())

    # lane_pos = traci.lanearea.getPosition()
    # if step >= 10:
    # #     # traci.close()
    #     traci.load(["-c", cfg_dir])
    #     for i in range(10):
    #         time.sleep(0.5)
    #         traci.simulationStep(i+1)
    #     break
traci.close(wait=False)
# sys.stdout.flush()
# traci.start([sumoBinary, "-c", cfg_dir])
# traci.load(["-c", cfg_dir])