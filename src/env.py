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
import SUMO_manager
from random_events_func import EventManager
from collections import deque
from pettingzoo import ParallelEnv  #bedziemy tego uzywac poniewaz biblioteka gymnasiium sama w sobie nie radzi sobie z wieloma agentami wiec musimy ja rozszerzyc

class SumoEnv(ParallelEnv):
    """Multi-agent PettingZoo environment wrapping a SUMO traffic simulation.

        Each agent controls the traffic light phase at one intersection.
        Observation: [car_counts(4), ambulances(4), buses(4), detector_status(4), pedestrians(4), sin_time, cos_time]
        Actions: 0 = NS green, 1 = WE green
        """

    metadata = {
        "render_modes": ["human"],
        "name": "sumo_krzyzak_v3"
    }

    # Traffic light phase indices (must match NetEdit TLS program)
    # Correct sequence: NS_GREEN → NS_YELLOW → ALL_RED → WE_GREEN → WE_YELLOW → ALL_RED
    PHASE_NS_GREEN = 0  # gGrrgGrrrGrG   (active green, NS)
    PHASE_NS_YELLOW = 1  # yyrryyrrrrrrr  (warning, 3 s)
    PHASE_ALL_RED_A = 2  # rrrrrrrrrrrr   (safety buffer after NS, 2 s)
    PHASE_WE_GREEN = 3  # rrgGrrrGGrGr   (active green, WE)
    PHASE_WE_YELLOW = 4  # rryyrryyrrrr   (warning, 3 s)
    PHASE_ALL_RED_B = 5  # rrrrrrrrrrrr   (safety buffer after WE, 2 s)

    # Phase durations in seconds
    YELLOW_DUR = 4.0
    ALLRED_DUR = 3.0
    GREEN_DUR = 10.0 # active green time per agent decision step

    # Reward weights
    W_HALTING = 1.0  # queued vehicles penalty
    W_JAM = 0.2  # jam length penalty
    W_OCC = 0.5  # occupancy penalty
    W_SWITCH = 2.0  # phase-switch penalty (discourages oscillation)
    W_BRAKING = 5.0  # emergency braking penalty
    W_PRIORITY = 1.0  # priority vehicle penalty/reward multiplier
    W_PED = 0.7  # pedestrian waiting penalty multiplier

    # Reward for priority vehicles that passed without waiting
    REWARD_AMBULANCE_PASS = 20.0
    REWARD_BUS_PASS = 10.0

    # Simulation step length — overwritten from SUMO after start_sim()
    _DEFAULT_SIM_STEP = 0.1

    def __init__(self, config_path: str, gui: bool = False, rank: int = 0):
        self.config_path = config_path
        self.gui = gui
        self.render_mode = "human" if gui else None

        self.possible_agents = ["J6", "J8", "J15"]
        self.agents = self.possible_agents[:]

        self.sumo = SUMO_manager.SumoManager(config_path, gui, rank=rank)
        self.events = EventManager(self.sumo)

        self._sim_step = self._DEFAULT_SIM_STEP
        self._update_step_counts()

        obs_shape = (35,)  # 4 cars + 4 ambulances + 4 buses + 4 status + 4 pedestrians + 2 time + 1 actual phase
        self.observation_spaces = {
            agent: spaces.Box(low=-2.0, high=100.0, shape=obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(2)
            for agent in self.possible_agents
        }

        self.last_action: dict[str, int] = {agent: 0 for agent in self.possible_agents}
        self.detector_history: dict[str, list[deque]] = {
            agent: [deque(maxlen=100) for _ in range(4)]
            for agent in self.possible_agents
        }

        self.episode_step = 0
        self.current_step = 0
        self.max_steps = 500
        self.target_phase: int | None = None  # set externally by CurriculumCallback
        self.neighbors_dict = self.sumo.check_neighbords()

    #PettingZoo API

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.sumo.close_sim()
        self.sumo.start_sim()

        # Fetch actual simulation step length and recompute loop counts
        self._sim_step = self.sumo.get_sim_step_duration()
        self._update_step_counts()

        self.agents = self.possible_agents[:]
        self.episode_step = 0
        self.current_step = self._initial_step_for_phase(options)

        for agent in self.agents:
            self.last_action[agent] = 0
            self.events.detector_status[agent] = [1, 1, 1, 1]

        return self._get_observation(), {agent: {} for agent in self.agents}

    def step(self, action: dict[str, int]):

        self.current_step += 1
        self.episode_step += 1

        # events
        self._generate_traffic()
        self.events.emergency_vehicle_deployment(probability=0.01) #emergency
        self._generate_buses()
        self.events.detector_malfunction()

        accumulated_priority_penalty = {agent:0 for agent in self.agents} #for every agent

        # --- Yellow phase (only for agents that changed action) ---
        for agent in self.agents:
            if action[agent] != self.last_action[agent]:
                yellow_phase = self.PHASE_NS_YELLOW if self.last_action[agent] == 0 else self.PHASE_WE_YELLOW
                self.sumo.set_traffic_light_phase(agent, yellow_phase)

        for _ in range(self._yellow_steps):
            self.sumo.simulation_step()
            for agent in self.agents:
                accumulated_priority_penalty[agent] += self._instant_priority_penalty(agent, self.last_action[agent])

        # --- All-red buffer (only for agents that changed action) ---
        for agent in self.agents:
            if action[agent] != self.last_action[agent]:
                allred_phase = self.PHASE_ALL_RED_A if self.last_action[agent] == 0 else self.PHASE_ALL_RED_B
                self.sumo.set_traffic_light_phase(agent, allred_phase)

        for _ in range(self._allred_steps):
            self.sumo.simulation_step()
            for agent in self.agents:
                accumulated_priority_penalty[agent] += self._instant_priority_penalty(agent,
                                                                                          self.last_action[agent])

        # --- Set new green phase ---
        for agent in self.agents:
            green_phase = self.PHASE_NS_GREEN if action[agent] == 0 else self.PHASE_WE_GREEN
            self.sumo.set_traffic_light_phase(agent, green_phase)

        # --- Active green phase ---
        for _ in range(self._green_steps):
            self.sumo.simulation_step()
            for agent in self.agents:
                accumulated_priority_penalty[agent] += self._instant_priority_penalty(agent, action[agent])

        observations = self._get_observation()
        rewards = {}
        terminated = {}
        truncated = {}

        is_simulation_empty = False
        is_time_up = self.episode_step >= self.max_steps

        for agent in self.agents:
            metrics = self.sumo.get_junction_metrics(agent)
            passing_bonus = self.sumo.get_gps_status(agent)
            action_changed = action[agent] != self.last_action[agent]

            rewards[agent] = self._get_reward(agent, metrics, action_changed,
                                              accumulated_priority_penalty[agent],
                                              passing_bonus)
            terminated[agent] = is_simulation_empty
            truncated[agent] = is_time_up

        self.last_action = dict(action)

        return observations, rewards, terminated, truncated, {agent: {} for agent in self.agents}

    def close(self) -> None:
        self.sumo.close_sim()

    # reward

    def _get_reward(
            self,
            agent: str,
            metrics: dict,
            action_changed: bool,
            priority_penalty: float,
            passing_bonus: list[dict],
    ) -> float:
        braking_penalty = self.sumo.get_emergency_stopping_count() * self.W_BRAKING
        switch_penalty = self.W_SWITCH if action_changed else 0.0
        ped_penalty = self._pedestrian_penalty(agent)

        priority_reward = sum(
            self.REWARD_AMBULANCE_PASS if v["type"] == "ambulance" else self.REWARD_BUS_PASS
            for v in passing_bonus
            if v["wait"] < 1.0 and v["type"] in {"ambulance", "city_bus"}
        )

        penalty = (
                self.W_HALTING * metrics["total_halting"]
                + self.W_JAM * metrics["max_jam_length"]
                + self.W_OCC * metrics["occupancy"]
                + switch_penalty
                + braking_penalty
        )
        bonus = (
                self.W_PRIORITY * priority_penalty
                + self.W_PRIORITY * priority_reward
                + self.W_PED * ped_penalty
        )
        return float(bonus - penalty)

    # Per-step priority penalty (called inside simulation loops)
    def _instant_priority_penalty(self, agent: str, current_action: int) -> float:
        """Returns a negative penalty if a priority vehicle is waiting at a red signal."""
        penalty = 0.0
        amb_presence = self.sumo.get_veh_presence("ambulance", agent)
        bus_presence = self.sumo.get_veh_presence("city_bus", agent)

        for idx in range(4):
            on_green = (idx < 2 and current_action == 0) or (idx >= 2 and current_action == 1)
            if not on_green:
                if amb_presence[idx]:
                    penalty -= 2.0
                if bus_presence[idx]:
                    penalty -= 0.6
        return penalty

    # observation

    def _get_observation(self) -> dict[str, np.ndarray]:
        time_of_day = self.events.get_normalized_time(self.current_step)
        sin_t = math.sin(2 * math.pi * time_of_day)
        cos_t = math.cos(2 * math.pi * time_of_day)

        observations = {}
        for agent in self.agents:
            car_counts = self._masked_detector_data(agent)
            ambulances = self.sumo.get_veh_presence("ambulance", agent)
            buses = self.sumo.get_veh_presence("city_bus", agent)
            status = self.events.detector_status[agent]
            pedestrians = self.sumo.get_pedestrian_presence(agent)
            neighbor_data=[]
            my_neighbors=self.neighbors_dict[agent] #pobieramy konkretnych sasiadow
            for neighbor_id in sorted(my_neighbors.keys()): # pobieramy tylko i wylacznie klucze
                edge=my_neighbors[neighbor_id] #pobieramy konkretna ulice
                presence=1.0
                cars=float(self.sumo.tc.edge.getLastStepVehicleNumber(edge))
                phase=float(self.last_action[neighbor_id])
                neighbor_data.extend([presence,cars,phase])

            while len(neighbor_data)<12:
                neighbor_data.append(0.0)

            current_phase = [float(self.last_action[agent])]

            observations[agent] = np.array(
                car_counts + ambulances + buses + status + pedestrians + [sin_t, cos_t] + current_phase + neighbor_data,
                dtype=np.float32,
            )
        return observations

    def _masked_detector_data(self, agent: str) -> list[float]:
        """Returns detector readings, substituting rolling average for broken sensors."""
        raw = self.sumo.get_detector_data(agent)
        status = self.events.detector_status[agent]
        result = []

        for i in range(4):
            if status[i] == 1:
                val = float(raw[i])
                self.detector_history[agent][i].append(val)
                result.append(val)
            else:
                history = self.detector_history[agent][i]
                result.append(sum(history) / len(history) if history else 0.0)
        return result

    # Pedestrian penalty

    def _pedestrian_penalty(self, agent: str) -> float:
        """Returns a negative penalty for each pedestrian held at a red crossing.

        Crossing geometry:
          idx 0,1 = N/S approaches → piesi przechodzą przez jezdnię EW
                    bezpieczne gdy auta NS jadą (action=0, EW red)
                    CZERWONE gdy WE zielone (action=1) → on_red gdy action==1

          idx 2,3 = E/W approaches → piesi przechodzą przez jezdnię NS
                    bezpieczne gdy auta WE jadą (action=1, NS red)
                    CZERWONE gdy NS zielone (action=0) → on_red gdy action==0
        """
        penalty = 0.0
        last_action = self.last_action[agent]

        for idx, is_waiting in enumerate(self.sumo.get_pedestrian_presence(agent)):
            if not is_waiting:
                continue
            on_red = (idx < 2 and last_action == 1) or (idx >= 2 and last_action == 0)
            if on_red:
                penalty -= 1.0
        return penalty

    # Traffic & bus generation

    def _generate_traffic(self) -> None:
        if not self.events.available_routes:
            self.events.find_routes()
        for route_id in self.events.available_routes:
            self.events.spawn_dynamic_traffic(self.current_step, route_id)

    def _generate_buses(self) -> None:
        bus_lines = [
            # (route_id,         stops,                                  line_name,        interval)
            ("route_WE", ["busStop_J6_East", "busStop_J15_East"], "100_WE", 300),
            ("route_EW", ["busStop_J15_West", "busStop_J6_West"], "100_EW", 300),
            ("route_NS_J6", ["busStop_J6_South"], "101_NS", 200),
            ("route_SN_J6", ["busStop_J6_North"], "101_SN", 210),
            ("route_NS_J8", ["busStop_J8_South"], "102_NS", 220),
            ("route_SN_J8", ["busStop_J8_North"], "102_SN", 230),
            ("route_NS_J15", ["busStop_J15_South"], "103_NS", 200),
            ("route_SN_J15", ["busStop_J15_North"], "103_SN", 210),
        ]
        for route_id, stops, line_name, interval in bus_lines:
            self.events.scheduled_bus_deployment(
                self.current_step, route_id, stops=stops,
                line_name=line_name, interval_steps=interval,
            )

    # Curriculum support

    def set_target_phase(self, phase: int | None) -> None:
        """Called by CurriculumCallback to pin the time-of-day phase for training."""
        self.target_phase = phase

    # Internal jelpers

    def _update_step_counts(self) -> None:
        """Recomputes simulation loop counts from the current step duration."""
        self._yellow_steps = round(self.YELLOW_DUR / self._sim_step)
        self._allred_steps = round(self.ALLRED_DUR / self._sim_step)
        self._green_steps = round(self.GREEN_DUR / self._sim_step)

    def _initial_step_for_phase(self, options: dict | None) -> int:
        """Returns a starting simulation step matching the curriculum time-of-day phase."""
        hour = EventManager.HOUR_STEP
        phase_ranges = {
            0: (0, 5 * hour),
            1: (5 * hour, 7 * hour),
            2: (7 * hour, 9 * hour),
            3: (9 * hour, 15 * hour),
            4: (15 * hour, 18 * hour),
            5: (18 * hour, 24 * hour),
        }
        phase = self.target_phase
        if phase is None and options:
            phase = options.get("target_phase")

        if phase is not None and phase in phase_ranges:
            lo, hi = phase_ranges[phase]
            return random.randint(lo, hi)

        return random.randint(0, 24 * hour)