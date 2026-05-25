import traci
import libsumo
from libsumo import junction
from networkx.classes import neighbors


class SumoManager:
    """Manages communication between the RL agent and the SUMO simulator via TraCI/libsumo."""

    TL_IDS = {
        "Kcynska": "cluster1876650944_300760980",
        "Zbozowa": "cluster300760946_300760949_300760970_300760977",
        "Owsiana": "cluster12708301500_12708314102_1309390847_1876644683_#2more"
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

    #random setup (we can change it later, but is needed for function)
    INTERSECTION_CONFIGS = {
        "Kcynska": {
            "num_phases": 10,
            "phase_length": [28, 5, 3, 6, 3, 28, 5, 3, 6, 3],
        },
        "Zbozowa": {
            "num_phases": 9,
            "phase_length": [32, 5, 3, 6, 3, 1, 32, 5, 3],

        },
        "Owsiana": {
            "num_phases": 10,
            "phase_length": [28, 5, 3, 6, 3, 28, 5, 3, 6, 3],
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
        """Sets the traffic light program phase using the mapped cluster ID."""
        # Używamy junction_id (np. "Kcynska") do pobrania ID klastra z XMLa
        real_tl_id = self.TL_IDS.get(junction_id)
        if not real_tl_id:
            print(f"[ERROR] Junction ID '{junction_id}' not found in TL_IDS!")
            return

        self._conn.trafficlight.setPhase(real_tl_id, phase)

        try:
            config=self.INTERSECTION_CONFIGS[junction_id]
            custom_duration=config["phase_length"][phase]

            self._conn.trafficlight.setPhaseDuration(real_tl_id, custom_duration)

        except KeyError:
            print("Warning -> No config for this intersection. Using default values.")
        except IndexError:
            print(f"Error-> {junction_id}  has not phase {phase}")





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
        for lane_idx in range(1, 5):  # Check if lanes exists from _1 to _4
            det_id = f"det_{junction_id}_{wlot}_{lane_idx}"
            try:
                if det_id in self._conn.lanearea.getIDList():
                    total += traci_func(det_id)
            except Exception:
                # No more lanes
                break
        return total

    def get_detector_data(self, junction_id: str) -> list[int]:
        """Returns vehicle counts per detector lane: e.g. [3, 2, 0, 5]."""
        return [
            int(self._get_aggregated_metric(junction_id,intake, self._conn.lanearea.getLastStepVehicleNumber))
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
        for idx, intake in enumerate(["1in", "2in", "3in", "4in"]):
            for lane_idx in range(1, 5):
                det_id = f"det_{junction_id}_{intake}_{lane_idx}"
                try:
                    for veh_id in self._conn.lanearea.getLastStepVehicleIDs(det_id):
                        if self._conn.vehicle.getTypeID(veh_id) == veh_type:
                            presence[idx] = 1
                            break
                    if presence[idx] == 1:
                        break
                except Exception:
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
        """Wykrywa sąsiednie skrzyżowania przez BFS po grafie krawędzi.

        Problem ze starym kodem: sprawdzał czy Kcynska_1out ∈ EXIT[Zbozowa],
        ale to dwie różne krawędzie połączone tylko przez wspólny węzeł pośredni.

        BFS rozwiązuje to przez:
          1. Budowę mapy węzeł → wychodzące krawędzie (z TraCI, raz na start_sim)
          2. Dla każdego skrzyżowania: BFS od jego krawędzi wyjściowych
          3. Gdy BFS natrafi na intake innego skrzyżowania → sąsiad znaleziony

        Zwraca: {junction: {neighbor: intake_edge_of_neighbor}}
        intake_edge_of_neighbor to krawędź tuż przed sąsiadem — idealna
        do monitorowania przepustowości między węzłami.
        """
        # Buduj graf: węzeł → [krawędzie wychodzące z tego węzła]
        node_to_outgoing: dict[str, list[str]] = {}
        for edge_id in self._conn.edge.getIDList():
            if edge_id.startswith(":"):  # pomiń wewnętrzne krawędzie SUMO
                continue
            from_node = self._conn.edge.getFromJunction(edge_id)
            node_to_outgoing.setdefault(from_node, []).append(edge_id)

        # Płaski słownik: intake_edge → junction_id  (do szybkiego sprawdzenia w BFS)
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
                        # Znalazłeś intake sąsiada — zapisz i nie idź dalej przez to skrzyżowanie
                        result[junc][neighbor] = next_edge
                    else:
                        # Krawędź pośrednia — dodaj do kolejki BFS
                        queue.append(next_edge)

        return result
        # Przykład wyniku:
        # {'Kcynska': {'Zbozowa': 'Zbozowa_1in'},
        #  'Zbozowa': {'Kcynska': 'Kcynska_Xin', 'Owsiana': 'Owsiana_1in'},
        #  'Owsiana': {'Zbozowa': 'Zbozowa_Xin'}}

























