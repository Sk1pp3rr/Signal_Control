# Step 1: Agent should get vector with the reflection of the world e.g. [3,0,0,0], it says that there are 3 cars in traffic jam on first line. spaces.Box?
# Step 2: Definition of action possible to be done by the agent, for 2 line intersection it would be just two actions with buffer action(yellow light).
# Agent does not have control on individual lights but only on phases defined in SUMO!!!
# Step 3: Cycle logic: Get action from AI (e.g. Agent chooses 0) -> Send command to SUMO by TraCI -> Do jump in time e.g. 5s -> check sensors -> Get the reward
# Step 4: Reward func for start kindergarden this will probably work: -(sum of cars in congestions on all detectors), Agent will try to minimalize his punishment
from __future__ import annotations

import math

from gymnasium import spaces
import numpy as np
import random
from core import SUMO_manager
from core.random_events_func import EventManager
from collections import deque
from pettingzoo import ParallelEnv


class SumoEnv(ParallelEnv):
    """Multi-agent PettingZoo environment wrapping a SUMO traffic simulation.

        Each agent controls the traffic light phase at one intersection.
        Observation: [car_counts(4), ambulances(4), buses(4), detector_status(4), pedestrians(4), sin_time, cos_time, current_phase(1), neighbor_data(12)]
        = 35 elementów łącznie
        Actions: 0 = NS green, 1 = WE green
        """

    metadata = {
        "render_modes": ["human"],
        "name": "sumo_krzyzak_v3"
    }

    PHASE_NS_GREEN = 0
    PHASE_NS_YELLOW = 1
    PHASE_ALL_RED_A = 2
    PHASE_WE_GREEN = 3
    PHASE_WE_YELLOW = 4
    PHASE_ALL_RED_B = 5

    YELLOW_DUR = 4.0
    ALLRED_DUR = 3.0
    GREEN_DUR = 10.0

    MIN_GREEN_STEPS = 3

    # Reward weights
    W_HALTING = 1.0
    W_JAM = 0.2
    W_OCC = 0.5
    W_SWITCH = 2.0
    W_BRAKING = 5.0
    W_PRIORITY = 1.0
    W_PED = 0.7

    REWARD_AMBULANCE_PASS = 20.0
    REWARD_BUS_PASS = 10.0

    _DEFAULT_SIM_STEP = 0.1

    INTERSECTION_CONFIGS = SUMO_manager.SumoManager.INTERSECTION_CONFIGS

    def __init__(self, config_path: str, gui: bool = False, rank: int = 0):

        self.config_path = config_path
        self.gui = gui
        self.render_mode = "human" if gui else None

        self.sumo = SUMO_manager.SumoManager(config_path, gui, rank=rank)
        self.events = EventManager(self.sumo)

        self.possible_agents = ["Kcynska", "Zbozowa", "Owsiana"]
        self.agents = self.possible_agents[:]

        self.episode_step = 0
        self.current_step = 0
        self.max_steps = 500

        self._sim_step = self._DEFAULT_SIM_STEP
        self._update_step_counts()

        obs_shape = (35,)
        self.observation_spaces = {
            agent: spaces.Box(low=-2.0, high=100.0, shape=obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(2) for agent in self.possible_agents
        }

        self.last_action: dict[str, int] = {agent: 0 for agent in self.possible_agents}
        self.steps_in_phase: dict[str, int] = {agent: 0 for agent in self.possible_agents}
        self.detector_history: dict[str, list[deque]] = {
            agent: [deque(maxlen=100) for _ in range(4)]
            for agent in self.possible_agents
        }

        self.target_phase: int | None = None

        self.sumo.start_sim()

        self._sim_step = self.sumo.get_sim_step_duration()
        self._update_step_counts()

        self.neighbors_dict = self.sumo.check_neighbors()
        self.ordered_neighbors_edges = {}
        for agent in self.agents:
            my_neighbors = self.neighbors_dict[agent]
            self.ordered_neighbors_edges[agent] = [
                (n_id, my_neighbors[n_id]) for n_id in sorted(my_neighbors.keys())
            ]

    # PettingZoo API

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

        self.agent_phase_counts = {
            agent: self.sumo.get_phase_count(agent) for agent in self.possible_agents
        }

        self._sim_step = self.sumo.get_sim_step_duration()
        self._update_step_counts()

        self.agents = self.possible_agents[:]
        self.episode_step = 0
        self.current_step = self._initial_step_for_phase(options)

        for agent in self.agents:
            self.last_action[agent] = 0
            self.steps_in_phase[agent] = 0
            self.events.detector_status[agent] = [1, 1, 1, 1]

        return self._get_observation(), {agent: {} for agent in self.agents}

    def step(self, action: dict[str, int]):
        self.current_step += 1
        self.episode_step += 1

        self._generate_traffic()
        self.events.emergency_vehicle_deployment(probability=0.01)
        self._generate_buses()
        self.events.detector_malfunction()

        accumulated_priority_penalty: dict[str, float] = {a: 0.0 for a in self.agents}

        effective_action: dict[str, int] = dict(action)
        for agent in self.agents:
            if (
                action[agent] != self.last_action[agent]
                and self.steps_in_phase[agent] < self.MIN_GREEN_STEPS
            ):
                effective_action[agent] = self.last_action[agent]
        action = effective_action

        changing = {a for a in self.agents if action[a] != self.last_action[a]}
        steady   = set(self.agents) - changing

        # Phase transition
        if changing:
            for agent in changing:
                cfg = self.INTERSECTION_CONFIGS[agent]
                trans_start = cfg["group_starts"][self.last_action[agent]] + 1
                self.sumo.set_traffic_light_phase(agent, trans_start)

            for agent in steady:
                cfg = self.INTERSECTION_CONFIGS[agent]
                self.sumo.set_traffic_light_phase(
                    agent, cfg["group_starts"][self.last_action[agent]]
                )

            max_trans = max(
                self._compute_transition_steps(a, self.last_action[a]) for a in changing
            )
            for _ in range(max_trans):
                self.sumo.simulation_step()

            for agent in self.agents:
                # FIX #3: _instant_priority_penalty zwraca wartości ujemne.
                # Akumulujemy je jako penalty (nie bonus) — odejmujemy w _get_reward.
                accumulated_priority_penalty[agent] += (
                    self._instant_priority_penalty(agent, self.last_action[agent]) * max_trans
                )

        # Active green
        for agent in self.agents:
            cfg = self.INTERSECTION_CONFIGS[agent]
            self.sumo.set_traffic_light_phase(agent, cfg["group_starts"][action[agent]])

        for _ in range(self._green_steps):
            self.sumo.simulation_step()

        for agent in self.agents:
            accumulated_priority_penalty[agent] += (
                self._instant_priority_penalty(agent, action[agent]) * self._green_steps
            )

        # Collect results
        observations = self._get_observation()
        rewards, terminated, truncated = {}, {}, {}
        is_time_up = self.episode_step >= self.max_steps

        for agent in self.agents:
            metrics       = self.sumo.get_junction_metrics(agent)
            passing_bonus = self.sumo.get_gps_status(agent)
            action_changed = agent in changing

            rewards[agent]    = self._get_reward(
                agent, metrics, action_changed,
                accumulated_priority_penalty[agent], passing_bonus,
            )
            terminated[agent] = False
            truncated[agent]  = is_time_up

        for agent in self.agents:
            if agent in changing:
                self.steps_in_phase[agent] = 0
            else:
                self.steps_in_phase[agent] += 1

        self.last_action = dict(action)
        return observations, rewards, terminated, truncated, {a: {} for a in self.agents}

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
                + self.W_PRIORITY * abs(priority_penalty)
        )
        bonus = (
                self.W_PRIORITY * priority_reward
                + ped_penalty
        )
        return float(bonus - penalty)

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
            neighbor_data = []
            for neighbor_id, edge in self.ordered_neighbors_edges[agent]:
                cars = float(self.sumo._conn.edge.getLastStepVehicleNumber(edge))
                phase = float(self.last_action[neighbor_id])
                neighbor_data.extend([1.0, cars, phase])

            while len(neighbor_data) < 12:
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
        """Returns a negative penalty for each pedestrian held at a red crossing."""
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

        spawn_attempts = 4 * len(self.events.INTAKES)
        for _ in range(spawn_attempts):
            self.events.spawn_dynamic_traffic(self.current_step)

    def _generate_buses(self) -> None:
        bus_lines = [
            ("route_1b", ["Kcynska02", "Zbozowa02", "Owsiana02"], "Bus_A", 200),
            ("route_2b", ["CisowaSibeliusa01", "Owsiana01", "Zbozowa01", "Kcynska01"], "Bus_B", 200),
            ("route_3b", ["Owsiana01", "Zbozowa01", "Kcynska01"], "Bus_C", 200),
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

    # Internal helpers

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

    def _compute_transition_steps(self, agent: str, last_action: int) -> int:
        """Oblicza liczbę kroków symulacji potrzebnych na przejście fazowe (żółta + all-red)."""
        cfg = self.INTERSECTION_CONFIGS[agent]
        group_start = cfg["group_starts"][last_action]
        phase_lengths = cfg["phase_length"]

        next_action = 1 - last_action  # przełączamy między 0 i 1
        next_green = cfg["group_starts"][next_action]

        total_duration = 0.0
        i = group_start + 1
        while i != next_green:
            total_duration += phase_lengths[i % len(phase_lengths)]
            i = (i + 1) % len(phase_lengths)

        return max(1, round(total_duration / self._sim_step))