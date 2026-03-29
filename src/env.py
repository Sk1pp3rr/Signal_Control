# Step 1: Agent should get vector with the reflection of the world e.g. [3,0,0,0], it says that there are 3 cars in traffic jam on first line. spaces.Box?
# Step 2: Definition of action possible to be done by the agent, for 2 line intersection it would be just two actions with buffer action(yellow light).
# Agent does not have control on individual lights but only on phases defined in SUMO!!!
# Step 3: Cycle logic: Get action from AI (e.g. Agent chooses 0) -> Send command to SUMO by TraCI -> Do jump in time e.g. 5s -> check sensors -> Get the reward
# Step 4: Reward func for start kindergarden this will probably work: -(sum of cars in congestions on all detectors), Agent will try to minimalize his punishment

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

class SumoEnv(gym.Env):
    def __init__(self,
                 config_path,
                 gui=False #weather we want to use GUI
                 ):
        #TODO: Definition of step 1 and 2
        #---Step 1: Observation space---
        #In krzyzak, we have four detectors, every one of them is giving number from 0 to 20
        self.observation_space = spaces.Box(
            low=0,
            high=20,
            shape=(4,),
            dtype=np.float32
        ) # box is the table of floats, it should be enough for AI to know where traffic is building
        #---Step 2: Action Space---
        # 0: vertical green light, 1: horizontally green light
        self.action_space = spaces.Discrete(2) #discrete action of possible "two buttons"
        self.config_path = config_path
        self.gui = gui

    def reset(self,
              seed=None,
              options = None
              ):
        #TODO: restart of SUMO env for nest trial

        #func to open TraCI

        initial_obs = np.zeros(4, dtype=np.float32)
        return initial_obs, {}

    def step(self,
             action #action performed by Agent
             ):
        #TODO: Definition of step 3 and 4 (action->Time->Reward)
        #1.Perform action
        #2.Move forward in time eg. 5s
        #3.Fetch new data from detectors
        #4.Get the reward

        obs = self._get_obs()
        reward = self._get_reward(obs)
        done = False  # If simulation ended

        return obs, reward, done, False, {}

    def _get_obs(self):
        #getter of observation data

        return np.zeros(4, dtype=np.float32)

    def _get_reward(self, obs):
        #mathematical evaluation of situation in SUMO

        reward = -float(np.sum(obs))
        return reward

    def close(self):
        #cleaning
        if traci.isLoaded():
            traci.close()
