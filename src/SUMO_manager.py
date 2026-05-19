import traci
import libsumo
from libsumo import junction


class SumoManager:
    """Manages communication between the RL agent and the SUMO simulator via TraCI/libsumo."""
    JUNCTION_DETECTORS = {
        "J6": ["e2_0", "e2_1", "e2_2", "e2_3"],
        "J8": ["e2_6", "e2_7", "e2_4", "e2_5"],
        "J15": ["e2_8", "e2_9", "e2_10", "e2_11"],
    }

    JUNCTION_INTAKE_EDGES = {
        "J6": ["-E6", "E3", "-E4", "-E5"],
        "J8": ["-E11", "E5", "-E10", "-E12"],
        "J15": ["-E15", "E12", "-E14", "-E13"],
    }

    JUNCTION_EXIT_EDGES = {
        "J6": ["E6", "-E3", "E4", "E5"],
        "J8": ["E11", "-E5", "E10", "E12"],
        "J15": ["E15", "-E12", "E14", "E13"],
    }

    PRIORITY_VEHICLE_TYPES = {"ambulance", "city_bus", "police", "fire_truck"}


    def __init__(self, config_path: str, gui: bool = False, rank: int = 0):
        self.config_path = config_path
        self.gui = gui
        self.rank = rank
        self.label = f"sim_{rank}"
        self.tc = traci if gui else libsumo
        self._sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", config_path, "--start"]


    # Internal helpers
    @property
    def _conn(self):
        """Returns the active TraCI/libsumo connection."""
        return self.tc.getConnection(self.label) if self.gui else self.tc

    # Simulation lifecycle

    def start_sim(self):
        """Start the SUMO simulator"""
        if self.gui:
            self.tc.start(self._sumo_cmd, label=self.label)
        else:
            self.tc.start(self._sumo_cmd)

    def close_sim(self):
        """Closes the running SUMO simulation, ignoring errors if already closed."""
        try:
            self._conn.close()
        except Exception:
            pass

    def simulation_step(self) -> None:
        """Advances the simulation by one step."""
        self._conn.simulationStep()

    def get_sim_step_duration(self) -> float:
        """Returns the simulation step length in seconds (e.g. 0.1 or 1.0)"""
        return self._conn.simulation.getDeltaT()

    # Traffic light control

    def set_traffic_light_phase(self, junction_id: str, phase: int) -> None:
        """Sets the traffic light program phase for the given junction."""
        self._conn.trafficlight.setPhase(junction_id, phase)

    # Detector and sensor data

    def get_detector_data(self, junction_id: str) -> list[int]:
        """Returns vehicle counts per detector lane: e.g. [3, 2, 0, 5]."""
        return [
            self._conn.lanearea.getLastStepVehicleNumber(det)
            for det in self.JUNCTION_DETECTORS[junction_id]
        ]

    def get_junction_metrics(self, junction_id: str) -> dict:
        """Returns aggregated traffic metrics (halting vehicles, jam length, occupancy)."""
        metrics = {"total_halting": 0, "max_jam_length": 0.0, "occupancy": 0.0}
        for det in self.JUNCTION_DETECTORS[junction_id]:
            metrics["total_halting"]  += self._conn.lanearea.getLastStepHaltingNumber(det)
            metrics["max_jam_length"] += self._conn.lanearea.getLastIntervalMaxJamLengthInMeters(det)
            metrics["occupancy"]      += self._conn.lanearea.getLastStepOccupancy(det)
        return metrics

    def get_veh_presence(self, veh_type: str, junction_id: str) -> list[int]:
        """Returns a binary presence vector [0/1] per detector lane for the given vehicle type."""
        presence = [0, 0, 0, 0]
        for idx, det_id in enumerate(self.JUNCTION_DETECTORS[junction_id]):
            for veh_id in self._conn.lanearea.getLastStepVehicleIDs(det_id):
                if self._conn.vehicle.getTypeID(veh_id) == veh_type:
                    presence[idx] = 1
                    break
        return presence

    def get_gps_status(self, junction_id: str) -> list[dict]:
        """Returns priority vehicles (ambulance, bus, etc.) that recently passed the junction.

        Simulates GPS integration: checks exit edges for low-distance vehicles,
        returning their type and accumulated waiting time.
        """
        passed_priority = []
        for edge_id in self.JUNCTION_EXIT_EDGES[junction_id]:
            for veh_id in self._conn.edge.getLastStepVehicleIDs(edge_id):
                if self._conn.vehicle.getDistance(veh_id) >= 10.0:
                    continue
                veh_type = self._conn.vehicle.getTypeID(veh_id)
                if veh_type in self.PRIORITY_VEHICLE_TYPES:
                    passed_priority.append({
                        "id": veh_id,
                        "type": veh_type,
                        "wait": self._conn.vehicle.getWaitingTime(veh_id),
                    })
        return passed_priority

    def get_pedestrian_presence(self, junction_id: str) -> list[float]:
        """Returns a binary presence vector [0.0/1.0] per intake edge for pedestrians.

        Simulates pedestrian crossing request buttons.
        """
        return [
            1.0 if self._conn.edge.getLastStepPersonIDs(edge) else 0.0
            for edge in self.JUNCTION_INTAKE_EDGES[junction_id]
        ]

    def get_emission_metrics(self, junction_id: str) -> tuple[float, float]:
        """Returns (total_fuel_consumption, total_CO2_emission) across all intake edges."""
        total_fuel = 0.0
        total_co2 = 0.0
        for edge_id in self.JUNCTION_INTAKE_EDGES[junction_id]:
            total_fuel += self._conn.edge.getFuelConsumption(edge_id)
            total_co2 += self._conn.edge.getCO2Emission(edge_id)
        return total_fuel, total_co2

    def get_emergency_stopping_count(self) -> int:
        """Returns the number of vehicles performing emergency stops this step."""
        return self._conn.simulation.getEmergencyStoppingVehiclesNumber()