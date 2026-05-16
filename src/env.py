# Step 1: Agent should get vector with the reflection of the world e.g. [3,0,0,0], it says that there are 3 cars in traffic jam on first line. spaces.Box?
# Step 2: Definition of action possible to be done by the agent, for 2 line intersection it would be just two actions with buffer action(yellow light).
# Agent does not have control on individual lights but only on phases defined in SUMO!!!
# Step 3: Cycle logic: Get action from AI (e.g. Agent chooses 0) -> Send command to SUMO by TraCI -> Do jump in time e.g. 5s -> check sensors -> Get the reward
# Step 4: Reward func for start kindergarden this will probably work: -(sum of cars in congestions on all detectors), Agent will try to minimalize his punishment
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import random
from tensorboard.compat.tensorflow_stub.tensor_shape import vector
import SUMO_manager
from random_events_func import eventManager
from collections import deque
from pettingzoo import ParallelEnv  #bedziemy tego uzywac poniewaz biblioteka gymnasiium sama w sobie nie radzi sobie z wieloma agentami wiec musimy ja rozszerzyc

class SumoEnv(ParallelEnv):
    def __init__(self,
                 config_path,
                 gui=False, #weather we want to use GUI
                 rank = 0
                 ):

        self.current_step=0
        self.episode_step=0
        self.max_steps = 500
        self._sim_step = 0.1 #zmienilem spowrotem na 0.1 zeby zapobiec warningom

        self.posible_agents=["J6", "J8", "J15"] #mozliwi agenci , jesli bedzie wiecej to sie doda poprzez petle
        self.agents=self.posible_agents[:]

        self.sumo = SUMO_manager.SumoManager(config_path, gui, rank = rank) #init of connector between Agent and SUMO
        self.events = eventManager(self.sumo)
        self.history_window = 100 #history of the last 100 correct readings
        self.detector_history = {
            agent: [deque(maxlen=self.history_window) for _ in range(4)]
            for agent in self.agents
        }
        self.last_action = {agent: 0 for agent in self.agents}
        #---Step 1: Observation space---

        #In krzyzak, we have four detectors, every one of them is giving number from 0 to 100
        self.observation_space ={agent:  spaces.Box(
            low=0,
            high=100,
            shape=(22,), #4 place for cars and 4 for ambulances and 4 for buses adn 4 for each detector status and 4 for pedestrians and 2 for normalized daytime sin and cos
            #it's done for time to be a cycle (avoiding jumps from 1.0 to 0)
            dtype=np.float32
        ) for agent in self.agents} # box is the table of floats, it should be enough for AI to know where traffic is building
        #---Step 2: Action Space---
        # 0: vertical green light, 1: horizontally green light
        self.action_space = {
            agent: spaces.Discrete(2) for agent in self.agents } #discrete action of possible "two buttons"
        self.last_action ={
            "J6": 0,
            "J8": 0,
            "J15": 0
        }  # last action performed by agent
        self.config_path = config_path
        self.gui = gui

        self.YELLOW_DUR = 4
        self.ALLRED_DUR = 2
        self.GREEN_DUR = 10

        self.PHASE_NS_GREEN = 0  # gGrrgGrrrGrG  32s
        self.PHASE_NS_YELLOW = 1  # yyrryyrrrrrrr  3s
        self.PHASE_ALL_RED_A = 2  # rrrrrrrrrrrr   2s  (bufor po NS)
        self.PHASE_WE_GREEN = 3  # rrgGrrrGGrGr  32s
        self.PHASE_WE_YELLOW = 4  # rryyrryyrrrr   3s
        self.PHASE_ALL_RED_B = 5  # rrrrrrrrrrrr   2s  (bufor po WE)

        # Time for steps
        self.yellow_steps = round(self.YELLOW_DUR / self._sim_step)  # 3s
        self.allred_steps = round(self.ALLRED_DUR / self._sim_step)  # 2s
        self.green_steps = round(self.GREEN_DUR / self._sim_step)  # 5s


    def reset(self,
              seed=None,
              options = None
              ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.sumo.close_sim() #If there were any simulations running close them.
        self.sumo.start_sim() #Start new simulation

        hour_step = 720
        self._choose_daytime(options,hour_step)

        self.episode_step = 0  # Zerujemy stoper epizodu

        for agent in self.agents:
            self.last_action[agent]=0
            self.events.detector_status[agent] = [1, 1, 1, 1]

        infos={agent:{} for agent in self.agents}

        return self._get_observation(), infos

    def step(self, action):

        conn = self.sumo.tc.getConnection(self.sumo.label) if self.gui else self.sumo.tc

        self.current_step += 1
        self.episode_step += 1
        #step_throughput = 0

        # events
        #Spawning cars in simulation steps to simulate traffic intensity throughout a day
        self._generate_traffic()

        self.events.emergnecy_vechicle_deployment(probability=0.01) #emergency

        #Buses
        self._generate_buses()

        self.events.detector_malfunction() #tutaj mamy jakas mozliwa

        accumulated_priority_penalty = {agent:0 for agent in self.agents} #for every agent

        for agent in self.agents:
            action_changed = action[agent] != self.last_action[agent]  # sprawdza
            if action_changed:
                if self.last_action[agent] == 0:
                   self.sumo.set_traffic_light_phase(agent, self.PHASE_NS_YELLOW)
                elif self.last_action[agent] == 1:
                   self.sumo.set_traffic_light_phase(agent, self.PHASE_WE_YELLOW)

        for _ in range(self.yellow_steps):
            conn.simulationStep()


            step_penalties=self._calculate_instant_priority_penalty(self.last_action)
            for agent in self.agents:
                accumulated_priority_penalty[agent] += step_penalties[agent]

        for agent in self.agents:
            action_changed = action[agent] != self.last_action[agent]  # sprawdza

            if action_changed:
               if self.last_action[agent] == 0:
                   self.sumo.set_traffic_light_phase(agent, self.PHASE_ALL_RED_A)
               elif self.last_action[agent] == 1:
                   self.sumo.set_traffic_light_phase(agent, self.PHASE_ALL_RED_B)


        for _ in range(self.allred_steps):
            conn.simulationStep()

            step_penalties=self._calculate_instant_priority_penalty(self.last_action)
            for agent in self.agents:
                accumulated_priority_penalty[agent] += step_penalties[agent]


        for agent in self.agents:
            if action[agent] == 0:
               self.sumo.set_traffic_light_phase(agent, self.PHASE_NS_GREEN)
            elif action[agent] == 1:
                self.sumo.set_traffic_light_phase(agent, self.PHASE_WE_GREEN)


        # Active green phase
        for _ in range(self.green_steps):
            conn.simulationStep()
            step_penalties=self._calculate_instant_priority_penalty(action)
            for agent in self.agents:
                accumulated_priority_penalty[agent] += step_penalties[agent]





        comb_obs=self._get_observation()
        reward = {}
        truncated={}
        terminated={}
        infos={agent:{} for agent in self.agents} #dla kazdego
        #is_simulation_empty = conn.simulation.getMinExpectedNumber() <= 0
        is_simulation_empty = False
        is_time_up=self.episode_step >= self.max_steps

        for agent in self.agents:
            metrics=self.sumo.get_junction_metrics(agent)
            pasing_bonus=self.sumo.get_gps_status(agent)
            agent_changed_action = (action[agent] != self.last_action[agent])
            reward[agent] = self._get_reward(metrics, agent_changed_action, accumulated_priority_penalty[agent], pasing_bonus,agent)
            terminated[agent] = is_simulation_empty
            truncated[agent] = is_time_up

        self.last_action = action

        return comb_obs, reward, terminated, truncated, infos





    def _calculate_instant_priority_penalty(self, current_action):
        """Penalty for every step in Simulation"""
        penalty_obs={}
        for agent in self.agents:
            penalty = 0.0
            amb_presence = self.sumo.get_veh_presence("ambulance", agent) #
            bus_presence = self.sumo.get_veh_presence("city_bus", agent)

            for idx in range(4):
                # Check if there is red in the intake of the junction
                is_red = not ((idx < 2 and current_action[agent] == 0) or (idx >= 2 and current_action[agent] == 1))
                if is_red:
                    if amb_presence[idx]:
                        penalty -= 2.0
                    if bus_presence[idx]:
                        penalty -= 0.6
            penalty_obs[agent] = penalty
        return penalty_obs

    #TODO: Implementation of reward for buses and everything with it
    def _get_reward(self,metrics, action_changed, priority_penalty, passing_bonus,agent):
        conn = self.sumo.tc.getConnection(self.sumo.label) if self.gui else self.sumo.tc
        #mathematical evaluation of situation in SUMO
        hc = 1 #multiplayer of queue_penalty for cars
        jc = 0.2 #multiplayer of waiting time penalty for cars
        oc = 0.5
        pp = 1.0
        pr = 1.0
        bc = 5.0
        ps = 0.7

        emergency_braking_count = conn.simulation.getEmergencyStoppingVehiclesNumber()
        braking_penalty = emergency_braking_count * bc

        #num of cars in queue
        halt_penalty = metrics['total_halting']
        occ_penalty = metrics['occupancy']
        jam_penalty = metrics['max_jam_length']
        ped_penalty = self.calculate_ped_penalty(agent)

        #switching penalty to avoid DISCO
        switch_penalty = 2.0 if action_changed else 0.0
        prior_reward = 0
        # rewards for smooth passage
        for vehicle in passing_bonus:
            if vehicle['wait'] < 1.0:
                if vehicle['type'] == "ambulance":
                    prior_reward += 20
                elif vehicle['type'] == "city_bus":
                    prior_reward += 10

        #full penalty
        reward = -(hc * halt_penalty + jc * jam_penalty + oc * occ_penalty + switch_penalty + braking_penalty) + pp * priority_penalty + pr * prior_reward + ped_penalty * ps

        return float(reward)

    def _get_observation(self):
        # Status 0 or 1

        observation={}
        time_of_day = self.events.get_normalized_time(self.current_step)  # agent should know what time of day it is

        sin_time = math.sin(2 * math.pi * time_of_day)
        cos_time = math.cos(2 * math.pi * time_of_day)

        for agent in self.agents:
            status = self.events.detector_status[agent]
            raw_data = self.sumo.get_detector_data(agent)


            masked_data = []
            # if status[i] == 0, put 0 masking
            for i in range(4):
                if status[i] == 1:
                    val = float(raw_data[i])
                    self.detector_history[agent][i].append(val)
                    masked_data.append(val)
                else:
                    if len(self.detector_history[agent][i]) > 0:
                        avg = sum(self.detector_history[agent][i])/len(self.detector_history[agent][i])
                        masked_data.append(float(avg))
                    else:
                        masked_data.append(0.0)

            #Rest od the sim
            ambulances = self.sumo.get_veh_presence(veh_type="ambulance", junction_id=agent)
            buses = self.sumo.get_veh_presence(veh_type="city_bus", junction_id=agent)
            pedestrians = self.sumo.get_pedestrian_presence(agent)

            observation[agent] = np.concatenate([masked_data, ambulances, buses, status, pedestrians, [sin_time, cos_time]]).astype(np.float32)
        # vector
        return observation

    def close(self):
        self.sumo.close_sim()


    def calculate_ped_penalty(self,agent):
        ped_presence = self.sumo.get_pedestrian_presence(agent)
        ped_penalty_accumulator = 0

        for idx, is_waiting in enumerate(ped_presence):
            if is_waiting:
                # Check if ped have red
                is_red_for_ped = ((idx < 2 and self.last_action[agent] == 0) or
                                  (idx >= 2 and self.last_action[agent] == 1))
                if is_red_for_ped:
                    ped_penalty_accumulator -= 1.0
        return ped_penalty_accumulator

    def _generate_buses(self):
        # Main lines going through
        self.events.scheduled_bus_deployment(
            self.current_step,
            route_id="route_WE",
            stops=["busStop_J6_East", "busStop_J15_East"],
            line_name="100_Express_WE",
            interval_steps=300
        )
        self.events.scheduled_bus_deployment(
            self.current_step,
            route_id="route_EW",
            stops=["busStop_J15_West", "busStop_J6_West"],
            line_name="100_Express_EW",
            interval_steps=300
        )

        # 2. Local J6 (West)
        self.events.scheduled_bus_deployment(
            self.current_step, "route_NS_J6",
            stops=["busStop_J6_South"],
            line_name="101_NS", interval_steps=200
        )
        self.events.scheduled_bus_deployment(
            self.current_step, "route_SN_J6",
            stops=["busStop_J6_North"],
            line_name="101_SN", interval_steps=210
        )

        # 3. Local J8 (Middle)
        self.events.scheduled_bus_deployment(
            self.current_step, "route_NS_J8",
            stops=["busStop_J8_South"],
            line_name="102_NS", interval_steps=220
        )
        self.events.scheduled_bus_deployment(
            self.current_step, "route_SN_J8",
            stops=["busStop_J8_North"],
            line_name="102_SN", interval_steps=230
        )

        # 4. Local J15 (East)
        self.events.scheduled_bus_deployment(
            self.current_step, "route_NS_J15",
            stops=["busStop_J15_South"],
            line_name="103_NS", interval_steps=200
        )
        self.events.scheduled_bus_deployment(
            self.current_step, "route_SN_J15",
            stops=["busStop_J15_North"],
            line_name="103_SN", interval_steps=210
        )

    def _generate_traffic(self):
        #load all roads
        if not self.events.available_routes:
            self.events.find_routes()
        #go through all roads
        for route_id in self.events.available_routes:
            self.events.spawn_dynamic_traffic(self.current_step, route_id)

    def _choose_daytime(self, options, hour_step = 720):
        if options and "target_phase" in options:
            phase = options["target_phase"]

            if phase == 0:  # Night (00:00 - 05:00)
                self.current_step = random.randint(0, 5 * hour_step)
            elif phase == 1:  # Early morning (05:00 - 07:00)
                self.current_step = random.randint(5 * hour_step, 7 * hour_step)
            elif phase == 2:  # Morning peak (07:00 - 09:00)
                self.current_step = random.randint(7 * hour_step, 9 * hour_step)
            elif phase == 3:  # Day (09:00 - 15:00)
                self.current_step = random.randint(9 * hour_step, 15 * hour_step)
            elif phase == 4:  # Afternoon peak (15:00 - 18:00)
                self.current_step = random.randint(15 * hour_step, 18 * hour_step)
            elif phase == 5:  # Evening (18:00 - 24:00)
                self.current_step = random.randint(18 * hour_step, 24 * hour_step)
        else:
            # Full random
            self.current_step = random.randint(0, 17280)