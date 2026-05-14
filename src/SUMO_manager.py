import traci
import libsumo
from libsumo import junction


class SumoManager:
    def __init__(self, config_path, gui=False, rank = 0):
        self.config_path = config_path
        self.sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", self.config_path, "--start"] #Manager will automatically work dependably of weather gui is enabled
        #self.DETECTORS = ["e2_0", "e2_1", "e2_2", "e2_3"] #table of detectors in "krzyzak"
        self.rank = rank
        self.label = f"sim_{self.rank}"
        self.gui = gui
        self.tc=traci if self.gui else libsumo
        self.junction_detectors={
            "J6": ["e2_0", "e2_1", "e2_2", "e2_3"],
            "J8": ["e2_6", "e2_7", "e2_4", "e2_5"],
            "J15": ["e2_8","e2_9","e2_10","e2_11"]
        } #bedziemy mieli tutaj detektory dla konretnego skrzyzowania

        self.junction_intake_edges={
            "J6": ["-E6", "E3", "-E4", "-E5"],
            "J8": ["-E11", "E5", "-E10", "-E12"],
            "J15": ["-E15", "E12", "-E14", "-E13"]
        }
        self.junction_exit_edges={
            "J6": ["E6","-E3","E4","E5"],
            "J8": ["E11","-E5","E10","E12"],
            "J15": ["E15","-E12","E14","E13"]
        }
        self.ped_edges = {
          "J6":  ["-E4", "E3", "-E5", "-E6"],
          "J7": ["-E4", "E3", "-E5", "-E6"]
        }



    def start_sim(self):
        """Start the SUMO simulator"""
        if self.gui: #dodane
            self.tc.start(self.sumo_cmd, label=self.label)
        else:
            self.tc.start(self.sumo_cmd)

    def close_sim(self):
        """Close the specific SUMO simulator"""
        try: #dodane
            if self.gui:
                conn=self.tc.getConnection(self.label)
                conn.close()
            else:
                self.tc.close() #dodane
        except Exception:
            pass
        
    def set_traffic_light_phase(self,junction_id, phase):
        """Set the traffic light phase on the specified junction_id"""
        if self.gui: #dodane
            conn = self.tc.getConnection(self.label)
            conn.trafficlight.setPhase(junction_id, phase)
        else:
            self.tc.trafficlight.setPhase(junction_id, phase)



    def get_detector_data(self, junction_id): #dodamy ze bedziemy pobierac z konkretnego skrzyzowania
        """Get the detection detector data"""
        detectors=self.junction_detectors[junction_id] #bierzemy konkretny detektor
        data = []
        conn=self.tc.getConnection(self.label) if self.gui else self.tc #dodane
        for detector in detectors:
            count = conn.lanearea.getLastStepVehicleNumber(detector) #get the data from individual detector
            data.append(count) # add them to our vector
        return data # e.g. [3,2,0,5]
    def get_waiting_time_data(self, junction_id):
        """Get the waiting time detector data"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc #dodane
        detectors=self.junction_detectors[junction_id] #bierzemy konkretny detektor
        total_waiting_time = 0
        for detector in detectors:
            # Get total weiting time from each generator
            total_waiting_time += conn.lanearea.getWaitingTime(detector)
        return total_waiting_time

    #def get_avg_waiting_time_data(self, junction_id): #tego nie uzywamy
        """Get the waiting time data (sum from vehicles on intake edges)"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc

        total_waiting_time = 0
        intake_edges = self.junction_intake_edges[junction_id]
        for edge_id in intake_edges:
            vehicles = conn.edge.getLastStepVehicleIDs(edge_id)
            for v_id in vehicles:
                total_waiting_time += conn.vehicle.getWaitingTime(v_id)
        return total_waiting_time

    def get_junction_metrics(self, junction_id):
        """Download raw data from SUMO junction"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc

        # Tworzymy NOWY słownik tylko dla tego jednego wywołania
        metrics = {
            'total_halting': 0,
            'max_jam_length': 0,
            'occupancy': 0
        }

        detectors = self.junction_detectors[junction_id]

        for detector in detectors:
            metrics['total_halting'] += conn.lanearea.getLastStepHaltingNumber(detector)
            metrics['max_jam_length'] += conn.lanearea.getLastIntervalMaxJamLengthInMeters(detector)
            metrics['occupancy'] += conn.lanearea.getLastStepOccupancy(detector)

        return metrics

    # This implementation is weird because I came to conclusion that not every ambulance will come from the "city of origin" where this agent could be working
    # sooo the agent can get the data both ways, not to neglect the existence of ambulances not integrated with the system.
    def get_veh_presence(self, veh_type, junction_id):
        """Returns vector [0,0,0,0] with 1, in place where there is veh_type (GPS or Sensor)"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc #dodane
        detectors=self.junction_detectors[junction_id]#bierzemy konkretny detektor
        presence=[0,0,0,0]
        for idx, det_id in enumerate(detectors):
            vehicle_on_det = conn.lanearea.getLastStepVehicleIDs(det_id)
            for veh_id in vehicle_on_det:
                if conn.vehicle.getTypeID(veh_id) == veh_type:
                    presence[idx] = 1
                    break
        return presence

    def get_gps_status(self, junction_id):
        """Simulates GPS data, return list od vehicles witch could be connected to the city network (potentially gps e.g. emergency and city_buses)
        witch left the simulation or rode through the junction (left the edges)"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc

        passed_priority = []

        exit_edges = self.junction_exit_edges[junction_id]

        for edge_id in exit_edges:
            # get all the car's id on intake edges
            vehicles = conn.edge.getLastStepVehicleIDs(edge_id)
            for v_id in vehicles:
                if conn.vehicle.getDistance(v_id) < 10.0:
                    v_type = conn.vehicle.getTypeID(v_id)
                    if v_type in ["ambulance","city_bus","police","fire_truck"]:
                        passed_priority.append({
                            'id': v_id,
                            'type': v_type,
                            'wait': conn.vehicle.getWaitingTime(v_id)
                        })
        return passed_priority

    def get_pedestrian_presence(self, junction_id):
        """Simulation of pedestrians buttons on crosswalks, return if they are any pedestrians on the edges"""
        ped_edges=self.ped_edges[junction_id]
        presence = []

        conn = self.tc.getConnection(self.label) if self.gui else self.tc #dodane

        for edge in ped_edges:
            # Check if there is more than 0 persons on the edge
            person_ids = conn.edge.getLastStepPersonIDs(edge)
            if len(person_ids) > 0:
                presence.append(1.0)  # Button pressed
            else:
                presence.append(0.0)

        return presence

    def get_emission_metrics(self,junction_id):
        """Gets data about CO2 emissions and fuel consumption on intakes of the junction"""
        total_fuel = 0
        total_co2 = 0
        intake_edges =self.junction_intake_edges[junction_id]

        conn = self.tc.getConnection(self.label) if self.gui else self.tc #dodane

        for edge_id in intake_edges:
            total_fuel += conn.edge.getFuelConsumption(edge_id)
            total_co2 += conn.edge.getCO2Emission(edge_id)

        return total_fuel, total_co2