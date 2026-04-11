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
            shape=(12,), #4 place for cars and 4 for ambulances and 4 detectors
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


        obs = self.sumo.get_detector_data()
        obs_amb=[0,0,0,0]
        self.events.detector_status=[1,1,1,1]
        det=self.events.detector_status

        initial_obs = np.array(obs+obs_amb+det, dtype=np.float32)
        return initial_obs, {}

    def step(self,
             action #action performed by Agent
             ):
        #1.Perform action
        #2.Move forward in time e.g. 5s
        #3.Fetch new data from detectors
        #4.Get the reward
        self.current_step += 1
        #Spawn of ambulance at the beginning of the step
        self.events.emergnecy_vechicle_deployment(probability=0.01)



        action_changed = action != self.last_action

        #if there were some action performed by agent use buffor of yellow light
        if action_changed:
            yellow_phase = 1 if self.last_action == 0 else 3
            self.sumo.set_traffic_light_phase("J6", yellow_phase)
            #skip for 3 minutes
            for _ in range(30): #3 sec for yellow
                traci.simulationStep()

        #phase switch
        phase = 0 if action == 0 else 2
        self.sumo.set_traffic_light_phase("J6", phase)

        amb_penalty_accumulator = self.simulate_and_get_ambulance_penalty(action, num_steps=50) #



        self.last_action = action # save last action

        self.events.detector_malfunction()  #

        detector_status = self.events.detector_status

        detector_data = self.is_detector_working(self.events.detector_status)

        ambulances=self.sumo.get_ambulance_presence()

        comb_obs = np.concatenate([detector_data,ambulances, detector_status]).astype(np.float32)

        metrics = self.sumo.get_junction_metrics()
        reward = self._get_reward(metrics, action_changed, amb_penalty_accumulator)

        truncated = self.current_step >= self.max_steps
        terminated = traci.simulation.getMinExpectedNumber() <= 0

        #gymnasium requires: obs, reward, terminated, truncated, info
        return comb_obs, reward, terminated, truncated, {}

    def _get_reward(self,metrics, action_changed, amb_penalty):
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
        reward = -(hc * halt_penalty + jc * jam_penalty + oc * occ_penalty + switch_penalty + amb_penalty)





        return float(reward)

    def close(self):
        #cleaning
        if traci.isLoaded():
            traci.close()

    def simulate_and_get_ambulance_penalty(self, action, num_steps=50):
        # same code as function
        amb_penalty_accumulator = 0

        for _ in range(num_steps):
            traci.simulationStep()

            amb_presence = self.sumo.get_ambulance_presence()

            for idx, is_amb in enumerate(amb_presence):
                if is_amb:
                    is_green = (idx < 2 and action == 0) or (idx >= 2 and action == 1)

                    if not is_green:
                        amb_penalty_accumulator += 10

        return amb_penalty_accumulator

    def is_detector_working(self, detector_status):
        raw_detector_data = self.sumo.get_detector_data()
        detector_data=[]
        for i in range(len(detector_status)):
            if detector_status[i] == 1:
                detector_data.append(raw_detector_data[i])
            else:
                detector_data.append(0)

        return detector_data




