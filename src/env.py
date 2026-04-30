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
class SumoEnv(gym.Env):
    def __init__(self,
                 config_path,
                 gui=False #weather we want to use GUI
                 ):
        #TODO: Definition of step 1 and 2

        self.current_step = 0
        self.max_steps = 500

        self.sumo = SUMO_manager.SumoManager(config_path, gui) #init of connector between Agent and SUMO
        self.events = eventManager(self.sumo)
        #---Step 1: Observation space---

        #In krzyzak, we have four detectors, every one of them is giving number from 0 to 100
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(12,), #4 place for cars and 4 for ambulances and 4 for buses
            dtype=np.float32
        ) # box is the table of floats, it should be enough for AI to know where traffic is building
        #---Step 2: Action Space---
        # 0: vertical green light, 1: horizontally green light
        self.action_space = spaces.Discrete(2) #discrete action of possible "two buttons"
        self.last_action = 0 # last action performed by agent
        self.config_path = config_path
        self.gui = gui

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

    def step(self,
             action #action performed by Agent
             ):
        #1.Perform action
        #2.Move forward in time e.g. 5s
        #3.Fetch new data from detectors
        #4.Get the reward
        self.current_step += 1
        action_changed = action != self.last_action

        #Spawn of ambulance at the beginning of the step
        self.events.emergnecy_vechicle_deployment(probability=0.01)
        #buses
        self.events.scheduled_bus_deployment(self.current_step, "route_NS",stops=["busStop_J6_South"],line_name = "101_A", interval_steps= 150)
        self.events.scheduled_bus_deployment(self.current_step, route_id="route_SN", stops=["busStop_J6_North"], line_name="101_B", interval_steps=180)
        self.events.scheduled_bus_deployment(self.current_step, route_id="route_WE", stops=["busStop_J6_West", "busStop_J6_East"],line_name="102", interval_steps=200)

        self.events.detector_malfunction()

        #acumulator of priority penalties/rewards
        accumulated_priority_penalty = 0.0

        #if there were some action performed by agent use buffor of yellow light
        if action_changed:
            yellow_phase = 1 if self.last_action == 0 else 3
            self.sumo.set_traffic_light_phase("J6", yellow_phase)
            #skip for 3 minutes
            for _ in range(30): #3 sec for yellow
                traci.simulationStep()
                accumulated_priority_penalty += self._calculate_instant_priority_penalty(action)

        #phase switch
        phase = 0 if action == 0 else 2
        self.sumo.set_traffic_light_phase("J6", phase)

        for _ in range(50):
            traci.simulationStep()
            accumulated_priority_penalty += self._calculate_instant_priority_penalty(action)

        self.last_action = action # save last action
        comb_obs = self._get_observation()
        metrics = self.sumo.get_junction_metrics()

        passing_bonus = self.sumo.get_gps_status()


        reward = self._get_reward(metrics, action_changed, accumulated_priority_penalty, passing_bonus)

        truncated = self.current_step >= self.max_steps
        terminated = traci.simulation.getMinExpectedNumber() <= 0

        #gymnasium requires: obs, reward, terminated, truncated, info
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
        #mathematical evaluation of situation in SUMO
        hc = 1 #multiplayer of queue_penalty for cars
        jc = 0.2 #multiplayer of waiting time penalty for cars
        oc = 0.5
        pp = 1.0
        pr = 1.0
        pb = 1.0

        #num of cars in queue
        halt_penalty = metrics['total_halting']
        occ_penalty = metrics['occupancy']
        jam_penalty = metrics['max_jam_length']

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
        reward = -(hc * halt_penalty + jc * jam_penalty + oc * occ_penalty + switch_penalty ) + pp * priority_penalty + pr * prior_reward

        return float(reward)

    def _get_observation(self):
        # Status 0 or 1
        status = self.events.detector_status
        raw_data = self.sumo.get_detector_data()

        # if status[i] == 0, put 0 masking
        masked_data = [raw_data[i] if status[i] == 1 else 0 for i in range(4)]

        #Rest od the sim
        ambulances = self.sumo.get_veh_presence(veh_type="ambulance")
        buses = self.sumo.get_veh_presence(veh_type="city_bus")

        # vector
        return np.concatenate([masked_data, ambulances, buses]).astype(np.float32)

    def close(self):
        #cleaning
        if traci.isLoaded():
            traci.close()

    def simulate_and_get_ambulance_penalty(self, action, num_steps=50):
        # same code as function
        amb_penalty_accumulator = 0

        for _ in range(num_steps):
            traci.simulationStep()

            amb_presence = self.sumo.get_veh_presence("ambulance")

            for idx, is_amb in enumerate(amb_presence):
                if is_amb:
                    is_green = (idx < 2 and action == 0) or (idx >= 2 and action == 1)

                    if not is_green:
                        amb_penalty_accumulator += 10

        return amb_penalty_accumulator