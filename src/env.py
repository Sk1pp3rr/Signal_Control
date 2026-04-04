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
import random_events_func
class SumoEnv(gym.Env):
    def __init__(self,
                 config_path,
                 gui=False #weather we want to use GUI
                 ):
        #TODO: Definition of step 1 and 2

        self.current_step = 0
        self.max_steps = 500
        self.event_manager = random_events_func.eventManager(self)
        self.sumo = SUMO_manager.SumoManager(config_path, gui) #init of connector between Agent and SUMO
        #---Step 1: Observation space---
        #In krzyzak, we have four detectors, every one of them is giving number from 0 to 20
        self.observation_space = spaces.Box(
            low=0,
            high=20,
            shape=(8,), #change for 8 parameters that agent get
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
        #TODO: restart of SUMO env for nest trial
        self.current_step = 0
        self.sumo.close_sim() #If there were any simulations running close them.
        self.sumo.start_sim() #Start new simulation
        self.last_action = 0

        data = self.sumo.get_detector_data()  # get 4 things from detectors

        ambulance_location = [0, 0, 0, 0]

        initial_obs = np.array(data + ambulance_location, dtype=np.float32)

        return initial_obs, {}

    def step(self,
             action #action performed by Agent
             ):
        #TODO: Definition of step 3 and 4 (action->Time->Reward)
        #1.Perform action
        #2.Move forward in time eg. 5s
        #3.Fetch new data from detectors
        #4.Get the reward
        self.current_step += 1
        self.event_manager.emergnecy_vechicle_deployment(probability=0.01) #chance for spawn ambulance
        action_changed = action != self.last_action

        #if there were some action performed by agent use buffor of yellow light
        if action != self.last_action:
            yellow_phase = 1 if self.last_action == 0 else 3
            self.sumo.set_traffic_light_phase("J6", yellow_phase)
            #skip for 3 minutes
            for _ in range(30):
                traci.simulationStep()
        #phase switch
        phase = 0 if action == 0 else 2
        self.sumo.set_traffic_light_phase("J6", phase)



        for _ in range(50):
            traci.simulationStep() # jump in time

        metrics = self.sumo.get_junction_metrics()

        self.last_action = action # save last action
        data=self.sumo.get_detector_data() #get 4 things from detectors
        is_ambulance_present, is_ambulance_stuck, ambulance_lane=self.sumo.get_ambulance_metrics()
        ambulance_location_vector = [0, 0, 0, 0]

        if is_ambulance_present:
            lane_mapping = ["E3_1", "E4_1", "E5_1", "E6_1"]

            if ambulance_lane in lane_mapping:
                idx = lane_mapping.index(ambulance_lane)
                ambulance_location_vector[idx] = 1

        combined_obs=data+ambulance_location_vector

        obs = np.array(combined_obs, dtype=np.float32)
        reward = self._get_reward(obs,action_changed,metrics,is_ambulance_present, is_ambulance_stuck)

        truncated = self.current_step >= self.max_steps
        terminated = traci.simulation.getMinExpectedNumber() <= 0

        #gymnasium requires: obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, {}

    def _get_reward(self, obs, action_changed,metrics, is_ambulance_present, is_ambulance_stuck):
        #mathematical evaluation of situation in SUMO
        hc = 1 #multiplayer of queue_penalty for cars
        jc = 0.2 #multiplayer of waiting time penalty for cars
        oc = 0.5

        #num of cars in queue
        halt_penalty = metrics['total_halting']
        occ_penalty = metrics['occupancy']
        jam_penalty = metrics['max_jam_length']

        #waiting time for all detectors
        # waiting_times = self.sumo.get_waiting_time_data()
        # waiting_penalty = np.sum(waiting_times)

        #switching penalty to avoid DISCO
        switch_penalty = 1.0 if action_changed else 0.0

        #full reward
        reward = -(hc * halt_penalty + jc * jam_penalty + oc * occ_penalty + switch_penalty)

        #ambulance logic

        if is_ambulance_present:
            if is_ambulance_stuck:
                reward -=500 #this situation cannot happen, therefore is enormouse punnishment
            else:
                reward -=20 #we punish agent for purpose that ambulance is present (why not)

        return float(reward)

    def close(self):
        #cleaning
        if traci.isLoaded():
            traci.close()
