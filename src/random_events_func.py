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
            0: 0.03,
            1: 0.10,
            2: 0.30,
            3: 0.15,
            4: 0.25,
            5: 0.10
    }


    def __init__(self, sumo_env):
        self.env = sumo_env
        self.available_routes: list[str] = []
        self.detector_status: dict[str, list[int]] = {
            "J6": [1,1,1,1],
            "J8": [1,1,1,1],
            "J15": [1,1,1,1]
        } #for every detector we have 4 status we can add more ofc

        self.detector_history: list[deque] = [
            deque(maxlen=100) for _ in range(4)
        ]

    #hellper func
    @property
    def _conn(self):
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

    # makes list of all avaliable routes defined in simulation
    def find_routes(self) -> None:
        """Caches all non-internal SUMO routes."""
        self.available_routes = [
            r for r in self._conn.route.getIDList()
            if not r.startswith("!")
        ]

    #traffic spawning

    def spawn_dynamic_traffic(self, current_step, route_id) -> None:
        """Spawns passenger cars and occasional heavy trucks based on time of day."""
        phase = self.get_time_phase(current_step)

        if random.random() < self.TRAFFIC_PROBS[phase]:
            veh_id = f"veh_{current_step}_{route_id}"
            # Using distribution from vTypeDistribution
            try:
                # Using distribution from vTypeDistribution
                self._conn.vehicle.add(veh_id, route_id, typeID="urban_cars")
            except traci.exceptions.TraCIException:
                # If car has a problem we don't want SUMO to break
                pass

        # Spawning bigger trucks only in the early morning
        if phase == 1 and random.random() < 0.015:
            truck_id = f"truck_{current_step}_{route_id}"
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
