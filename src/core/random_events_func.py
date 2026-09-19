from __future__ import annotations

import traci
import random

from setuptools import Extension
from collections import deque

class EventManager:
    '''Manages random and scheduled simulation events: traffic spawning, bus lines,
     emergency vehicles, and detector malfunction.'''
    MALFUNCTION_PROB = 0.0003

    REPAIR_PROB = 0.01

    HOUR_STEP = 720

    TRAFFIC_PROBS = {
            0: 0.15,
            1: 0.40,
            2: 0.90,
            3: 0.60,
            4: 0.85,
            5: 0.40
    }

    INTAKES = ["-286967103", "169121911#1", "-27398735", "-177211203#1", "Kcynska_3in", "27398644#1", "53533699",
                    "27399957#0"]
    OUTPUTS = ["136687277#2", "180623102#1", "Owsiana_1out", "27398735", "177211203#0", "-27399957#5",
                    "Kcynska_2out", "Kcynska_4out", "-53533699"]
    INTAKE_WEIGHTS = {
        "-286967103": 0.08,
        "169121911#1": 0.25,
        "-27398735": 0.08,
        "27399957#0": 0.04,
        "-177211203#1": 0.04,
        "Kcynska_3in": 0.08,
        "27398644#1": 0.25,
        "53533699": 0.08
    }

    OUTPUTS_WEIGHTS = {
        "136687277#2":0.08,
        "180623102#1":0.08,
        "Owsiana_1out":0.25,
        "27398735":0.08,
        "177211203#0":0.08,
        "-27399957#5":0.08,
        "Kcynska_2out":0.08,
        "Kcynska_4out":0.25,
        "-53533699":0.08
    }

    DEFAULT_WEIGHT = 0.05


    def __init__(self, sumo_env):
        self.env = sumo_env
        self.available_routes: list[str] = []
        self.detector_status: dict[str, list[int]] = {
            "Kcynska": [1,1,1,1],
            "Zbozowa": [1,1,1,1],
            "Owsiana": [1,1,1,1]
        } #for every detector we have 4 status we can add more ofc

        # self.detector_history: list[deque] = [
        #     deque(maxlen=100) for _ in range(4)
        # ]
        self.detector_history: dict[str, list[deque]] = {
            junction: [deque(maxlen=100) for _ in range(4)]
            for junction in self.detector_status
        }
        self.veh_counter = 0

    #hellper func
    @property
    def _conn(self):
        """Returns the active TraCI / libsumo connection."""
        return self.env.tc.getConnection(self.env.label) if self.env.gui else self.env.tc

    #Daytime utils

    def get_time_phase(self,current_step) -> int:
        """Maps a simulation step to a time-of-day phase index (0–5)."""
        day_time = current_step % (24 * self.HOUR_STEP)

        thresholds = [
            (5 * self.HOUR_STEP, 0),  # Night (00:00 - 05:00)
            (7 * self.HOUR_STEP, 1),  # Early morning (05:00 - 07:00)
            (9 * self.HOUR_STEP, 2),  # Morning peak (07:00 - 09:00)
            (15 * self.HOUR_STEP, 3),  # Day (09:00 - 15:00)
            (18 * self.HOUR_STEP, 4)  # Afternoon peak (15:00 - 18:00)
        ]
        for threshold, phase in thresholds:
            if day_time < threshold:
                return phase
        return 5  # Evening (18:00 - 00:00)

    def get_normalized_time(self, current_step) -> float:
        """Returns the fraction of the day elapsed (0.0 – 1.0) for cyclic time encoding."""
        day_steps = 24 * self.HOUR_STEP
        return (current_step % day_steps) / day_steps

    #Route discovery

    def find_routes(self) -> None:
        """Loads all pre-defined routes from the running simulation and
        computes a combined weight for each one.

        Weight = INTAKE_WEIGHTS[first_edge] × OUTPUT_WEIGHTS[last_edge].
        Edges not present in the weight dicts receive DEFAULT_WEIGHT, so
        every route participates — just with lower probability.
        Call this once after start_sim(), then again after each reset.
        """
        self.available_routes = [
            r for r in self._conn.route.getIDList()
            if not r.startswith("!")
        ]
        self._build_route_weights()

    def _build_route_weights(self) -> None:
        """Computes and caches the selection weight for every loaded route."""
        self._route_weights = {}
        for route_id in self.available_routes:
            try:
                edges = self._conn.route.getEdges(route_id)
            except Exception:
                continue
            if not edges:
                continue
            w_in = self.INTAKE_WEIGHTS.get(edges[0], self.DEFAULT_WEIGHT)
            w_out = self.OUTPUTS_WEIGHTS.get(edges[-1], self.DEFAULT_WEIGHT)
            self._route_weights[route_id] = w_in * w_out


    #traffic spawning

    def spawn_dynamic_traffic(self, current_step: int) -> None:
        """Spawns one vehicle this step using weighted route selection.

        Selection logic:
          1. Gate on time-of-day probability (TRAFFIC_PROBS).
          2. Pick a route with probability of INTAKE_WEIGHTS[start] × OUTPUT_WEIGHTS[end].
          3. Spawn an urban_cars vehicle on that route.
          4. During early morning, also attempt to spawn a heavy_truck
             (small probability, same route).

        No route computation is done here — all routes come from the
        pre-defined .rou.xml file already loaded by find_routes().
        """
        if not self._route_weights:
            return

        phase = self.get_time_phase(current_step)
        if random.random() >= self.TRAFFIC_PROBS[phase]:
            return

        route_id = random.choices(
            list(self._route_weights.keys()),
            weights=list(self._route_weights.values()),
            k=1,
        )[0]

        self.veh_counter += 1
        veh_id = f"veh_{current_step}_{self.veh_counter}"
        try:
            self._conn.vehicle.add(veh_id, route_id, typeID="urban_cars")
        except traci.exceptions.TraCIException:
            pass

        # Heavy truck: early-morning only, low base probability
        if phase == 1 and random.random() < 0.015:
            truck_id = f"truck_{current_step}_{random.randint(0, 9999)}"
            try:
                self._conn.vehicle.add(truck_id, route_id, typeID="heavy_truck")
            except traci.exceptions.TraCIException:
                pass

    def scheduled_bus_deployment(
            self,
            current_step: int,
            route_id: str,
            stops: list[str] | None = None,
            line_name: str = "101",
            interval_steps: int = 100,
    ) -> None:
        """Deploys a bus on a fixed schedule with optional stop assignments.

        Args:
            current_step:   Current simulation step.
            route_id:       SUMO route ID for the bus.
            stops:          List of bus stop IDs where the bus should halt.
            line_name:      Displayed line number (e.g. "101_A").
            interval_steps: How many steps between successive bus deployments.
        """
        if current_step == 0 or current_step % interval_steps != 0:
            return

        veh_id = f"bus_{line_name}_{current_step}"
        try:
            self._conn.vehicle.add(veh_id, route_id, typeID="city_bus")
            self._conn.vehicle.setLine(veh_id, line_name)
            for stop_id in (stops or []):
                self._conn.vehicle.setBusStop(veh_id, stop_id, duration=20)
        except traci.exceptions.TraCIException:
            pass

    def emergency_vehicle_deployment(self, probability: float = 0.001) -> None:
        """Randomly spawns an emergency vehicle (ambulance) on a random route."""
        if random.random() >= probability:
            return

        if not self.available_routes:
            self.find_routes()

        route_id = random.choice(self.available_routes)
        veh_id = f"emergency_{self._conn.simulation.getTime()}"
        try:
            self._conn.vehicle.add(veh_id, route_id, typeID="ambulance")
            self._conn.vehicle.setColor(veh_id, (255, 0, 0, 255))
        except Exception as exc:
            print(f"[EventManager] Failed to spawn emergency vehicle: {exc}")


    #Detector malfunctions

    def detector_malfunction(self) -> None:
        """Randomly breaks working detectors and repairs broken ones each step."""
        for agent_id, status_list in self.detector_status.items():
            for i, status in enumerate(status_list):
                if status == 1 and random.random() < self.MALFUNCTION_PROB:
                    self.detector_status[agent_id][i] = 0
                    print(f"[EventManager] Detector malfunction on {agent_id}, lane {i}")
                elif status == 0 and random.random() < self.REPAIR_PROB:
                    self.detector_status[agent_id][i] = 1
                    print(f"[EventManager] Detector repaired on {agent_id}, lane {i}")

    #unused in actual phase

    def collision(self):
        conn = self.env.tc.getConnection(self.env.label) if self.env.gui else self.env.tc
        if not self.available_routes:
            self.find_routes()
        # Random choice of lane
        target_lane = random.choice(self.available_routes)
        conn.lane.setDisallowed(target_lane, ["passenger", "bus", "truck"])
        return