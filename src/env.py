# Step 1: Agent should get vector with the reflection of the world e.g. [3,0,0,0], it says that there are 3 cars in traffic jam on first line. spaces.Box?
# Step 2: Definition of action possible to be done by the agent, for 2 line intersection it would be just two actions with buffer action(yellow light).
# Agent does not have control on individual lights but only on phases defined in SUMO!!!
# Step 3: Cycle logic: Get action from AI (e.g. Agent chooses 0) -> Send command to SUMO by TraCI -> Do jump in time e.g. 5s -> check sensors -> Get the reward
# Step 4: Reward func for start kindergarden this will probably work: -(sum of cars in congestions on all detectors), Agent will try to minimalize his punishment

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import SUMO_manager
from random_events_func import eventManager
from collections import deque

class SumoEnv(gym.Env):
    def __init__(self,
                 config_path,
                 gui=False, #weather we want to use GUI
                 rank = 0
                 ):
        #TODO: Definition of step 1 and 2

        self.current_step = 0
        self.max_steps = 500
        self._sim_step = 0.1

        self.sumo = SUMO_manager.SumoManager(config_path, gui, rank = rank) #init of connector between Agent and SUMO
        self.events = eventManager(self.sumo)

        self.history_window = 100 #history of the last 100 correct readings
        self.detector_history = [deque(maxlen=self.history_window) for _ in range(4)]
        #---Step 1: Observation space---

        #In krzyzak, we have four detectors, every one of them is giving number from 0 to 100
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(20,), #4 place for cars and 4 for ambulances and 4 for buses adn 4 for each detector status and 4 for pedestrians
            dtype=np.float32
        ) # box is the table of floats, it should be enough for AI to know where traffic is building
        #---Step 2: Action Space---
        # 0: vertical green light, 1: horizontally green light
        self.action_space = spaces.Discrete(2) #discrete action of possible "two buttons"
        self.last_action = 0 # last action performed by agent
        self.config_path = config_path
        self.gui = gui

        self.YELLOW_DUR = 3
        self.ALLRED_DUR = 3
        self.GREEN_DUR = 5

        self.PHASE_NS_GREEN = 0  # gGrrgGrrrGrG  32s
        self.PHASE_NS_YELLOW = 1  # yyrryyrrrrrrr  3s
        self.PHASE_ALL_RED_A = 2  # rrrrrrrrrrrr   2s  (bufor po NS)
        self.PHASE_WE_GREEN = 3  # rrgGrrrGGrGr  32s
        self.PHASE_WE_YELLOW = 4  # rryyrryyrrrr   3s
        self.PHASE_ALL_RED_B = 5  # rrrrrrrrrrrr   2s  (bufor po WE)

    def reset(self,
              seed=None,
              options = None
              ):
        self.current_step = 0
        self.sumo.close_sim() #If there were any simulations running close them.
        self.sumo.start_sim() #Start new simulation
        self.last_action = 0


        detector_data = self.sumo.get_detector_data()
        ambulances = self.sumo.get_veh_presence("ambulance")
        buses = self.sumo.get_veh_presence("city_bus")

        self.events.detector_status = [1, 1, 1, 1]
        return self._get_observation(), {}

    def step(self, action):

        conn = traci.getConnection(self.sumo.label)

        self.current_step += 1
        action_changed = action != self.last_action

        #step_throughput = 0



        # events
        self.events.emergnecy_vechicle_deployment(probability=0.01)
        self.events.scheduled_bus_deployment(self.current_step, "route_NS",
                                             stops=["busStop_J6_South"], line_name="101_A", interval_steps=150)
        self.events.scheduled_bus_deployment(self.current_step, route_id="route_SN",
                                             stops=["busStop_J6_North"], line_name="101_B", interval_steps=180)
        self.events.scheduled_bus_deployment(self.current_step, route_id="route_WE",
                                             stops=["busStop_J6_East"], line_name="102", interval_steps=200)
        self.events.scheduled_bus_deployment(self.current_step, route_id="route_EW",
                                             stops=["busStop_J6_West"], line_name="103", interval_steps=200)

        self.events.detector_malfunction()

        accumulated_priority_penalty = 0.0

        # Time for steps
        yellow_steps = round(self.YELLOW_DUR / self._sim_step)  # 3s
        allred_steps = round(self.ALLRED_DUR / self._sim_step)  # 2s
        green_steps = round(self.GREEN_DUR / self._sim_step)  # 5s aktywnej fazy

        if action_changed:
            if self.last_action == 0:
                # Change: NS green (faza 0) → WE green (Phase 3)
                # Step 1: yellow NS
                self.sumo.set_traffic_light_phase("J6", self.PHASE_NS_YELLOW)
                for _ in range(yellow_steps):
                    conn.simulationStep()
                    accumulated_priority_penalty += self._calculate_instant_priority_penalty(self.last_action)
                    #step_throughput += conn.simulation.getArrivedNumber()

                # Step 2 2: All-red
                self.sumo.set_traffic_light_phase("J6", self.PHASE_ALL_RED_A)
                for _ in range(allred_steps):
                    conn.simulationStep()
                    accumulated_priority_penalty += self._calculate_instant_priority_penalty(self.last_action)
                    #step_throughput += conn.simulation.getArrivedNumber()

                # Step 3: WE green
                self.sumo.set_traffic_light_phase("J6", self.PHASE_WE_GREEN)

            else:
                # Change: WE green (faza 3) → NS green (faza 0)
                # Step 1: yellow WE
                self.sumo.set_traffic_light_phase("J6", self.PHASE_WE_YELLOW)
                for _ in range(yellow_steps):
                    conn.simulationStep()
                    accumulated_priority_penalty += self._calculate_instant_priority_penalty(self.last_action)
                    #step_throughput += conn.simulation.getArrivedNumber()

                # Step 2: All-red
                self.sumo.set_traffic_light_phase("J6", self.PHASE_ALL_RED_B)
                for _ in range(allred_steps):
                    conn.simulationStep()
                    accumulated_priority_penalty += self._calculate_instant_priority_penalty(self.last_action)
                    #step_throughput += conn.simulation.getArrivedNumber()

                # Step 3: NS green
                self.sumo.set_traffic_light_phase("J6", self.PHASE_NS_GREEN)

        else:
            # No change — Check if phase is correct
            target_phase = self.PHASE_NS_GREEN if action == 0 else self.PHASE_WE_GREEN
            self.sumo.set_traffic_light_phase("J6", target_phase)

        # Active green phase
        for _ in range(green_steps):
            conn.simulationStep()
            accumulated_priority_penalty += self._calculate_instant_priority_penalty(action)
            #step_throughput += conn.simulation.getArrivedNumber()

        self.last_action = action
        comb_obs = self._get_observation()
        metrics = self.sumo.get_junction_metrics()
        passing_bonus = self.sumo.get_gps_status()
        reward = self._get_reward(metrics, action_changed, accumulated_priority_penalty, passing_bonus)

        truncated = self.current_step >= self.max_steps
        terminated = conn.simulation.getMinExpectedNumber() <= 0

        #info = {'step_throughput': step_throughput}

        return comb_obs, reward, terminated, truncated, {}

    def _calculate_instant_priority_penalty(self, current_action):
        """Penalty for every step in Simulation"""
        penalty = 0.0
        amb_presence = self.sumo.get_veh_presence("ambulance")
        bus_presence = self.sumo.get_veh_presence("city_bus")

        for idx in range(4):
            # Check if there is red in the intake of the junction
            is_red = not ((idx < 2 and current_action == 0) or (idx >= 2 and current_action == 1))
            if is_red:
                if amb_presence[idx]:
                    penalty -= 2.0
                if bus_presence[idx]:
                    penalty -= 0.6
        return penalty

    #TODO: Implementation of reward for buses and everything with it
    def _get_reward(self,metrics, action_changed, priority_penalty, passing_bonus):
        conn = traci.getConnection(self.sumo.label)
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
        ped_penalty = self.calculate_ped_penalty()

        #switching penalty to avoid DISCO
        switch_penalty = 2.0 if action_changed else 0.0
        prior_reward = 0
        # rewards for smooth passage
        passed = self.sumo.get_gps_status()
        for vehicle in passed:
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
        status = self.events.detector_status
        raw_data = self.sumo.get_detector_data()

        masked_data = []
        # if status[i] == 0, put 0 masking
        for i in range(4):
            if status[i] == 1:
                val = float(raw_data[i])
                self.detector_history[i].append(val)
                masked_data.append(val)
            else:
                if len(self.detector_history[i]) > 0:
                    avg = sum(self.detector_history[i])/len(self.detector_history[i])
                    masked_data.append(float(avg))
                else:
                    masked_data.append(0.0)

        #Rest od the sim
        ambulances = self.sumo.get_veh_presence(veh_type="ambulance")
        buses = self.sumo.get_veh_presence(veh_type="city_bus")
        pedestrians = self.sumo.get_pedestrian_presence()

        # vector
        return np.concatenate([masked_data, ambulances, buses, status, pedestrians]).astype(np.float32)

    def close(self):
        self.sumo.close_sim()

    def simulate_and_get_ambulance_penalty(self, action, num_steps=50):
        conn = traci.getConnection(self.sumo.label)
        # same code as function
        amb_penalty_accumulator = 0

        for _ in range(num_steps):
            conn.simulationStep()

            amb_presence = self.sumo.get_veh_presence("ambulance")

            for idx, is_amb in enumerate(amb_presence):
                if is_amb:
                    is_green = (idx < 2 and action == 0) or (idx >= 2 and action == 1)

                    if not is_green:
                        amb_penalty_accumulator += 10

        return amb_penalty_accumulator

    def calculate_ped_penalty(self):
        ped_presence = self.sumo.get_pedestrian_presence()
        ped_penalty_accumulator = 0

        for idx, is_waiting in enumerate(ped_presence):
            if is_waiting:
                # Check if ped have red
                is_red_for_ped = ((idx < 2 and self.last_action == 0) or
                                  (idx >= 2 and self.last_action == 1))
                if is_red_for_ped:
                    ped_penalty_accumulator -= 1.0
        return ped_penalty_accumulator
