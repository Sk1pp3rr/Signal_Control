import traci
import libsumo
from libsumo import junction
from networkx.classes import neighbors


class SumoManager:
    """Manages communication between the RL agent and the SUMO simulator via TraCI/libsumo."""

    TL_IDS = {
        "Kcynska": "cluster1876650944_300760980",
        "Zbozowa": "cluster12708301500_12708314102_1309390847_1876644683_#2more",
        "Owsiana": "cluster300760946_300760949_300760970_300760977"
    }

    JUNCTION_INTAKE_EDGES = {
        "Kcynska": ["Kcynska_1in", "Kcynska_2in", "Kcynska_3in", "Kcynska_4in"],
        "Zbozowa": ["Zbozowa_1in", "Zbozowa_2in", "Zbozowa_3in", "Zbozowa_4in"],
        "Owsiana": ["Owsiana_1in", "Owsiana_2in", "Owsiana_3in", "Owsiana_4in"],
    }

    JUNCTION_EXIT_EDGES = {
        "Kcynska": ["Kcynska_1out", "Kcynska_2out", "Kcynska_3out", "Kcynska_4out"],
        "Zbozowa": ["Zbozowa_1out", "Zbozowa_2out", "Zbozowa_3out", "Zbozowa_4out"],
        "Owsiana": ["Owsiana_1out", "Owsiana_2out", "Owsiana_3out", "Owsiana_4out"],
    }

    PRIORITY_VEHICLE_TYPES = {"ambulance", "city_bus", "police", "fire_truck"}

    TARGET_GREEN_PHASES = {
        "Kcynska": {0: 0, 1: 5},
        "Zbozowa": {0: 0, 1: 6},
        "Owsiana": {0: 0, 1: 5}
    }

    # FIX #1: Dodano "group_starts" do każdego skrzyżowania.
    # group_starts[akcja] = indeks fazy zielonej dla danej akcji (0=NS, 1=WE).
    # trans_start = group_starts[akcja] + 1 to faza przejściowa (żółta/all-red).
    # Wartości muszą zgadzać się z programem TLS w pliku XML SUMO!
    INTERSECTION_CONFIGS = {
        "Kcynska": {
            "num_phases": 10,
            "phase_length": [28, 5, 3, 6, 3, 28, 5, 3, 6, 3],
            "group_starts": {0: 0, 1: 5},  # akcja 0 → faza 0 (NS green), akcja 1 → faza 5 (WE green)
        },
        "Zbozowa": {
            "num_phases": 9,
            "phase_length": [32, 5, 3, 6, 3, 1, 32, 5, 3],
            "group_starts": {0: 0, 1: 6},  # akcja 0 → faza 0 (NS green), akcja 1 → faza 6 (WE green)
        },
        "Owsiana": {
            "num_phases": 10,
            "phase_length": [28, 5, 3, 6, 3, 28, 5, 3, 6, 3],
            "group_starts": {0: 0, 1: 5},  # akcja 0 → faza 0 (NS green), akcja 1 → faza 5 (WE green)
        }
    }


    def __init__(self, config_path: str, gui: bool = False, rank: int = 0):
        self.config_path = config_path
        self.gui = gui
        self.rank = rank
        self.label = f"sim_{rank}"
        self.tc = traci if gui else libsumo
        self._sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", config_path, "--start"]
        self.posible_agents = ["Kcynska", "Zbozowa", "Owsiana"]
        self.agents = self.posible_agents[:]


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

        # --- Agresywny RAM Cache detektorów ---
        all_dets = set(self._conn.lanearea.getIDList())
        self.detector_cache = {}
        for j_id in self.agents:  # Używamy self.agents (posible_agents)
            self.detector_cache[j_id] = {}
            for wlot in ["1in", "2in", "3in", "4in"]:
                valid_list = []
                for lane_idx in range(1, 5):
                    det_id = f"det_{j_id}_{wlot}_{lane_idx}"
                    if det_id in all_dets:
                        valid_list.append(det_id)
                self.detector_cache[j_id][wlot] = valid_list

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
    def get_current_phase(self, junction_id: str) -> int:
        """Zwraca obecny indeks fazy dla danego skrzyżowania."""
        real_tl_id = self.TL_IDS.get(junction_id)
        return self._conn.trafficlight.getPhase(real_tl_id)

    def advance_to_next_phase(self, junction_id: str) -> None:
        """Popycha program świateł o jedną fazę do przodu i ustawia jej czas z konfigu."""
        real_tl_id = self.TL_IDS.get(junction_id)
        current_phase = self._conn.trafficlight.getPhase(real_tl_id)
        max_phases = self.get_phase_count(junction_id)

        next_phase = (current_phase + 1) % max_phases
        self._conn.trafficlight.setPhase(real_tl_id, next_phase)

        try:
            duration = self.INTERSECTION_CONFIGS[junction_id]["phase_length"][next_phase]
            self._conn.trafficlight.setPhaseDuration(real_tl_id, float(duration))
        except (KeyError, IndexError):
            pass

    def set_traffic_light_phase(self, junction_id: str, phase: int) -> None:
        """Sets the traffic light program phase using the mapped cluster ID."""
        real_tl_id = self.TL_IDS.get(junction_id)
        if not real_tl_id:
            print(f"[ERROR] Junction ID '{junction_id}' not found in TL_IDS!")
            return

        self._conn.trafficlight.setPhase(real_tl_id, phase)

        try:
            config = self.INTERSECTION_CONFIGS[junction_id]
            custom_duration = config["phase_length"][phase]
            self._conn.trafficlight.setPhaseDuration(real_tl_id, custom_duration)
        except KeyError:
            print("Warning -> No config for this intersection. Using default values.")
        except IndexError:
            print(f"Error-> {junction_id} has not phase {phase}")

    def get_phase_count(self, junction_id: str) -> int:
        """Num of phase using self._conn and TL_IDS."""
        try:
            if self._conn:
                logic_defs = self._conn.trafficlight.getCompleteRedYellowGreenDefinition(self.TL_IDS[junction_id])
                if logic_defs:
                    return len(logic_defs[0].phases)
        except Exception:
            pass
        return 8

    # Detector and sensor data

    def _get_aggregated_metric(self, junction_id: str, wlot: str, traci_func) -> float:
        """Sums data from detectors in one lane (except pedestrian line _0)"""
        total = 0.0
        for det_id in self.detector_cache[junction_id][wlot]:
            total += traci_func(det_id)
        return total

    def get_detector_data(self, junction_id: str) -> list[int]:
        """Returns vehicle counts per detector lane: e.g. [3, 2, 0, 5]."""
        return [
            int(self._get_aggregated_metric(junction_id, intake, self._conn.lanearea.getLastStepVehicleNumber))
            for intake in ["1in", "2in", "3in", "4in"]
        ]

    def get_junction_metrics(self, junction_id: str) -> dict:
        """Returns aggregated traffic metrics (halting vehicles, jam length, occupancy)."""
        metrics = {"total_halting": 0, "max_jam_length": 0.0, "occupancy": 0.0}
        for intake in ["1in", "2in", "3in", "4in"]:
            metrics["total_halting"]  += self._get_aggregated_metric(junction_id, intake, self._conn.lanearea.getLastStepHaltingNumber)
            metrics["max_jam_length"] += self._get_aggregated_metric(junction_id, intake, self._conn.lanearea.getJamLengthMeters)
            metrics["occupancy"]      += self._get_aggregated_metric(junction_id, intake, self._conn.lanearea.getLastStepOccupancy)
        return metrics

    def get_veh_presence(self, veh_type: str, junction_id: str) -> list[int]:
        """Returns a binary presence vector [0/1] per detector lane for the given vehicle type."""
        presence = [0, 0, 0, 0]
        veh_prefix = "emergency" if veh_type == "ambulance" else "bus"

        for idx, intake in enumerate(["1in", "2in", "3in", "4in"]):
            for det_id in self.detector_cache[junction_id][intake]:
                for veh_id in self._conn.lanearea.getLastStepVehicleIDs(det_id):
                    if veh_prefix in veh_id:
                        presence[idx] = 1
                        break
                if presence[idx] == 1:
                    break
        return presence

    def get_gps_status(self, junction_id: str) -> list[dict]:
        passed_priority = []
        for edge_id in self.JUNCTION_EXIT_EDGES[junction_id]:
            for veh_id in self._conn.edge.getLastStepVehicleIDs(edge_id):
                is_amb = "emergency" in veh_id
                is_bus = "bus" in veh_id

                if not (is_amb or is_bus):
                    continue

                if self._conn.vehicle.getDistance(veh_id) >= 10.0:
                    continue

                passed_priority.append({
                    "id": veh_id,
                    "type": "ambulance" if is_amb else "city_bus",
                    "wait": self._conn.vehicle.getWaitingTime(veh_id),
                })
        return passed_priority

    def get_pedestrian_presence(self, junction_id: str) -> list[float]:
        """Returns a binary presence vector [0.0/1.0] per intake edge for pedestrians."""
        presence = []
        for edge in self.JUNCTION_INTAKE_EDGES[junction_id]:
            try:
                presence.append(1.0 if self._conn.edge.getLastStepPersonIDs(edge) else 0.0)
            except Exception:
                presence.append(0.0)
        return presence

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

    def check_neighbors(self) -> dict[str, dict[str, str]]:
        """Wykrywa sąsiednie skrzyżowania przez BFS po grafie krawędzi."""
        node_to_outgoing: dict[str, list[str]] = {}
        for edge_id in self._conn.edge.getIDList():
            if edge_id.startswith(":"):
                continue
            from_node = self._conn.edge.getFromJunction(edge_id)
            node_to_outgoing.setdefault(from_node, []).append(edge_id)

        intake_of: dict[str, str] = {
            edge: junc
            for junc, edges in self.JUNCTION_INTAKE_EDGES.items()
            for edge in edges
        }

        result: dict[str, dict[str, str]] = {junc: {} for junc in self.agents}

        for junc in self.agents:
            visited: set[str] = set(self.JUNCTION_EXIT_EDGES[junc])
            queue: list[str] = list(self.JUNCTION_EXIT_EDGES[junc])

            while queue:
                edge = queue.pop(0)
                to_node = self._conn.edge.getToJunction(edge)

                for next_edge in node_to_outgoing.get(to_node, []):
                    if next_edge in visited:
                        continue
                    visited.add(next_edge)

                    neighbor = intake_of.get(next_edge)
                    if neighbor is not None and neighbor != junc:
                        result[junc][neighbor] = next_edge
                    else:
                        queue.append(next_edge)

        return result