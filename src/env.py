import math
import random
from collections import deque

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

import SUMO_manager
from random_events_func import EventManager


class SumoEnv(ParallelEnv):
    """Multi-agent PettingZoo environment for SUMO traffic light optimisation.

    Each agent controls one traffic-light junction.  The action space is
    Discrete(2): action 0 keeps/activates phase-group 0 (intakes 1in + 4in),
    action 1 keeps/activates phase-group 1 (intakes 2in + 3in).

    Observation per agent (35 floats):
        car_counts(4) · ambulances(4) · buses(4) · detector_status(4) ·
        pedestrians(4) · sin_time · cos_time · current_group(1) ·
        neighbor_data(12, zero-padded)
    """

    metadata = {"render_modes": ["human"], "name": "sumo_morska_v1"}

    # ------------------------------------------------------------------
    # Per-junction TLS configuration
    #
    # group_starts   – phase indices where each green group begins
    # green_intakes  – which intake indices (0-3) are active per group
    #                  (derived from net.xml link analysis)
    # ------------------------------------------------------------------
    INTERSECTION_CONFIGS: dict[str, dict] = {
        "Kcynska": {
            "num_phases":    10,
            "phase_length":  [28, 5, 5, 6, 5, 28, 5, 5, 6, 5],
            "group_starts":  [0, 5],
            "green_intakes": [{0, 3}, {1, 2}],
        },
        "Zbozowa": {
            "num_phases":    9,
            "phase_length":  [32, 5, 5, 6, 5, 1, 32, 5, 5],
            "group_starts":  [0, 6],
            "green_intakes": [{0, 3}, {1, 2}],
        },
        "Owsiana": {
            "num_phases":    10,
            "phase_length":  [28, 5, 5, 6, 5, 28, 5, 5, 6, 5],
            "group_starts":  [0, 5],
            "green_intakes": [{0, 3}, {1, 2}],
        },
    }

    # Active-green duration per agent decision step (seconds)
    GREEN_DUR = 10.0

    # Reward weights
    W_HALTING  = 1.0
    W_JAM      = 0.2
    W_OCC      = 0.5
    W_SWITCH   = 2.0
    W_BRAKING  = 5.0
    W_PRIORITY = 1.0
    W_PED      = 0.7

    REWARD_AMBULANCE_PASS = 20.0
    REWARD_BUS_PASS       = 10.0

    _DEFAULT_SIM_STEP = 0.1

    # Init

    def __init__(self, config_path: str, gui: bool = False, rank: int = 0) -> None:
        self.config_path = config_path
        self.gui         = gui
        self.render_mode = "human" if gui else None

        self.possible_agents = ["Kcynska", "Zbozowa", "Owsiana"]
        self.agents          = self.possible_agents[:]

        self.sumo   = SUMO_manager.SumoManager(config_path, gui, rank=rank)
        self.events = EventManager(self.sumo)

        self._sim_step = self._DEFAULT_SIM_STEP
        self._green_steps = round(self.GREEN_DUR / self._sim_step)

        obs_shape = (35,)
        self.observation_spaces = {
            a: spaces.Box(low=-2.0, high=100.0, shape=obs_shape, dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(2) for a in self.possible_agents
        }

        self.last_action: dict[str, int] = {a: 0 for a in self.possible_agents}
        self.detector_history: dict[str, list[deque]] = {
            a: [deque(maxlen=100) for _ in range(4)] for a in self.possible_agents
        }

        # Populated in reset() after start_sim() — safe default until then
        self.ordered_neighbors_edges: dict[str, list[tuple[str, str]]] = {
            a: [] for a in self.possible_agents
        }

        self.episode_step  = 0
        self.current_step  = 0
        self.max_steps     = 500
        self.target_phase: int | None = None

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

        self._sim_step    = self.sumo.get_sim_step_duration()
        self._green_steps = round(self.GREEN_DUR / self._sim_step)

        # Build neighbor map (requires running simulation for TraCI calls)
        neighbor_map = self.sumo.check_neighbors()
        for agent in self.possible_agents:
            neighbors = neighbor_map.get(agent, {})
            self.ordered_neighbors_edges[agent] = [
                (n_id, edge) for n_id, edge in sorted(neighbors.items())
            ]

        self.agents       = self.possible_agents[:]
        self.episode_step = 0
        self.current_step = self._initial_step_for_phase(options)

        for agent in self.agents:
            self.last_action[agent]               = 0
            self.events.detector_status[agent]    = [1, 1, 1, 1]
            self.events.available_routes.clear()

        self.events.find_routes()

        return self._get_observation(), {a: {} for a in self.agents}

    # Step

    def step(self, action: dict[str, int]):
        self.current_step += 1
        self.episode_step += 1

        self._generate_traffic()
        self.events.emergency_vehicle_deployment(probability=0.01)
        self._generate_buses()
        self.events.detector_malfunction()

        accumulated_priority_penalty: dict[str, float] = {a: 0.0 for a in self.agents}

        changing = {a for a in self.agents if action[a] != self.last_action[a]}
        steady   = set(self.agents) - changing

        # Phase transition
        if changing:
            for agent in changing:
                cfg         = self.INTERSECTION_CONFIGS[agent]
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

        self.last_action = dict(action)
        return observations, rewards, terminated, truncated, {a: {} for a in self.agents}

    def close(self) -> None:
        self.sumo.close_sim()

    # Reward

    def _get_reward(
        self,
        agent: str,
        metrics: dict,
        action_changed: bool,
        priority_penalty: float,
        passing_bonus: list[dict],
    ) -> float:
        braking_penalty = self.sumo.get_emergency_stopping_count() * self.W_BRAKING
        switch_penalty  = self.W_SWITCH if action_changed else 0.0
        ped_penalty     = self._pedestrian_penalty(agent)

        priority_reward = sum(
            self.REWARD_AMBULANCE_PASS if v["type"] == "ambulance" else self.REWARD_BUS_PASS
            for v in passing_bonus
            if v["wait"] < 1.0 and v["type"] in {"ambulance", "city_bus"}
        )

        penalty = (
            self.W_HALTING * metrics["total_halting"]
            + self.W_JAM   * metrics["max_jam_length"]
            + self.W_OCC   * metrics["occupancy"]
            + switch_penalty
            + braking_penalty
        )
        bonus = (
            self.W_PRIORITY * priority_penalty
            + self.W_PRIORITY * priority_reward
            + self.W_PED     * ped_penalty
        )
        return float(bonus - penalty)

    def _instant_priority_penalty(self, agent: str, current_action: int) -> float:
        """Negative penalty for each priority vehicle waiting at a red intake."""
        green   = self.INTERSECTION_CONFIGS[agent]["green_intakes"][current_action]
        amb     = self.sumo.get_veh_presence("ambulance", agent)
        bus     = self.sumo.get_veh_presence("city_bus",  agent)
        penalty = 0.0
        for idx in range(4):
            if idx not in green:
                if amb[idx]: penalty -= 2.0
                if bus[idx]: penalty -= 0.6
        return penalty

    def _pedestrian_penalty(self, agent: str) -> float:
        """Negative penalty for each pedestrian waiting at a red crossing."""
        green   = self.INTERSECTION_CONFIGS[agent]["green_intakes"][self.last_action[agent]]
        penalty = 0.0
        for idx, waiting in enumerate(self.sumo.get_pedestrian_presence(agent)):
            if waiting and idx not in green:
                penalty -= 1.0
        return penalty

    # Observation

    def _get_observation(self) -> dict[str, np.ndarray]:
        t     = self.events.get_normalized_time(self.current_step)
        sin_t = math.sin(2 * math.pi * t)
        cos_t = math.cos(2 * math.pi * t)

        observations = {}
        for agent in self.agents:
            car_counts  = self._masked_detector_data(agent)
            ambulances  = self.sumo.get_veh_presence("ambulance", agent)
            buses       = self.sumo.get_veh_presence("city_bus",  agent)
            status      = self.events.detector_status[agent]
            pedestrians = self.sumo.get_pedestrian_presence(agent)

            neighbor_data: list[float] = []
            for neighbor_id, edge in self.ordered_neighbors_edges[agent]:
                cars  = float(self.sumo.get_edge_vehicle_count(edge))
                phase = float(self.last_action[neighbor_id])
                neighbor_data.extend([1.0, cars, phase])
            while len(neighbor_data) < 12:
                neighbor_data.append(0.0)

            observations[agent] = np.array(
                car_counts + ambulances + buses + status + pedestrians
                + [sin_t, cos_t, float(self.last_action[agent])]
                + neighbor_data,
                dtype=np.float32,
            )
        return observations

    def _masked_detector_data(self, agent: str) -> list[float]:
        raw    = self.sumo.get_detector_data(agent)
        status = self.events.detector_status[agent]
        result = []
        for i in range(4):
            if status[i] == 1:
                val = float(raw[i])
                self.detector_history[agent][i].append(val)
                result.append(val)
            else:
                hist = self.detector_history[agent][i]
                result.append(sum(hist) / len(hist) if hist else 0.0)
        return result

    # Traffic generation

    def _generate_traffic(self) -> None:
        if not self.events.available_routes:
            self.events.find_routes()
        for _ in range(10 * len(self.events.INTAKE_WEIGHTS)):
            self.events.spawn_dynamic_traffic(self.current_step)

    def _generate_buses(self) -> None:
        bus_lines = [
            ("route_1b", ["Kcynska02", "Zbozowa02", "Owsiana02"],                          "Bus_A", 200),
            ("route_2b", ["CisowaSibeliusa01", "Owsiana01", "Zbozowa01", "Kcynska01"],     "Bus_B", 200),
            ("route_3b", ["Owsiana01", "Zbozowa01", "Kcynska01"],                          "Bus_C", 200),
        ]
        for route_id, stops, line_name, interval in bus_lines:
            self.events.scheduled_bus_deployment(
                self.current_step, route_id, stops=stops,
                line_name=line_name, interval_steps=interval,
            )

    # Curriculum support

    def set_target_phase(self, phase: int | None) -> None:
        self.target_phase = phase

    # Internal helpers

    def _compute_transition_steps(self, junction: str, from_group: int) -> int:
        """Simulation steps needed to run the transition out of *from_group*.

        Sums the phase_lengths of all phases between the current group's
        main-green and the next group's main-green, then converts to steps.

        Handles wrap-around (last group → first group).
        """
        cfg        = self.INTERSECTION_CONFIGS[junction]
        starts     = cfg["group_starts"]
        lengths    = cfg["phase_length"]
        n          = cfg["num_phases"]
        cur_start  = starts[from_group]
        nxt_start  = starts[(from_group + 1) % len(starts)]

        if nxt_start > cur_start:
            trans_range = range(cur_start + 1, nxt_start)
        else:
            trans_range = list(range(cur_start + 1, n)) + list(range(0, nxt_start))

        total_s = sum(lengths[i] for i in trans_range)
        return round(total_s / self._sim_step)

    def _initial_step_for_phase(self, options: dict | None) -> int:
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