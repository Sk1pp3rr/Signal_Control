import traci
import libsumo


class SumoManager:
    def __init__(self, config_path, gui=False, rank = 0):
        self.config_path = config_path
        self.sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", self.config_path, "--start"] #Manager will automatically work dependably of weather gui is enabled
        self.DETECTORS = ["e2_0", "e2_1", "e2_2", "e2_3"] #table of detectors in "krzyzak"
        self.rank = rank
        self.label = f"sim_{self.rank}"
        self.gui = gui
        self.tc=traci if self.gui else libsumo

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
        except self.tc.exceptions.TraCIException:
            pass
        except self.tc.exceptions.FatalTraCIError:
            pass

    def set_traffic_light_phase(self,junction_id, phase):
        """Set the traffic light phase on the specified junction_id"""
        if self.gui: #dodane
            conn = self.tc.getConnection(self.label)
            conn.trafficlight.setPhase(junction_id, phase)
        else:
            self.tc.trafficlight.setPhase(junction_id, phase)


    def get_detector_data(self):
        """Get the detection detector data"""
        data = []
        conn=self.tc.getConnection(self.label) if self.gui else self.tc #dodane
        for detector in self.DETECTORS:
            count = conn.lanearea.getLastStepVehicleNumber(detector) #get the data from individual detector
            data.append(count) # add them to our vector
        return data # e.g. [3,2,0,5]

    def get_waiting_time_data(self):
        """Get the waiting time detector data"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc #dodane

        total_waiting_time = 0
        for detector in self.DETECTORS:
            # Get total weiting time from each generator
            total_waiting_time += conn.lanearea.getWaitingTime(detector)
        return total_waiting_time

    def get_avg_waiting_time_data(self):
        """Get the waiting time data (sum from vehicles on intake edges)"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc

        total_waiting_time = 0
        intake_edges = ["-E6", "E3", "-E4", "-E5"]
        for edge_id in intake_edges:
            vehicles = conn.edge.getLastStepVehicleIDs(edge_id)
            for v_id in vehicles:
                total_waiting_time += conn.vehicle.getWaitingTime(v_id)
        return total_waiting_time

    def get_junction_metrics(self):
        """Download raw data from SUMO junction"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc #dodane

        metrics = {
            'total_halting': 0,
            'max_jam_length': 0,
            'occupancy': 0
        }
        for detector in self.DETECTORS:

            metrics['total_halting'] += conn.lanearea.getLastStepHaltingNumber(detector)
            metrics['max_jam_length'] += conn.lanearea.getLastIntervalMaxJamLengthInMeters(detector)
            metrics['occupancy'] += conn.lanearea.getLastStepOccupancy(detector)
        return metrics

     #funkja sprawdza czy mamy ambulans na mapie lub czy utknela w korku
    def get_ambulance_metrics(self):
        conn = self.tc.getConnection(self.label) if self.gui else self.tc #dodane

        vehicle_id=conn.vehicle.getIDList()
        ambulances=[] # were prepering list for ambulances
        for i in vehicle_id:
            if conn.vehicle.getTypeID(i) == "ambulance":
                is_ambulance=True
                is_ambulance_stuck = False
                ambulance_lane = None

                waiting_time=conn.vehicle.getWaitingTime(i) #we check how long ambulance are waiting
                if waiting_time > 5:
                    is_ambulance_stuck=True
                for detector in self.DETECTORS:
                    cars_in_detecor=conn.lanearea.getLastStepVehicleIDs(detector) #list of cars that are in range of detector
                    if i in cars_in_detecor:
                        ambulance_lane=detector
                wektor=[is_ambulance,is_ambulance_stuck, ambulance_lane]
                ambulances.append(wektor)


        return ambulances

    # This implementation is weird because I came to conclusion that not every ambulance will come from the "city of origin" where this agent could be working
    # sooo the agent can get the data both ways, not to neglect the existence of ambulances not integrated with the system.
    def get_veh_presence(self, veh_type):
        """Returns vector [0,0,0,0] with 1, in place where there is veh_type (GPS or Sensor)"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc #dodane

        presence=[0,0,0,0]
        for idx, det_id in enumerate(self.DETECTORS):
            vehicle_on_det = conn.lanearea.getLastStepVehicleIDs(det_id)
            for veh_id in vehicle_on_det:
                if conn.vehicle.getTypeID(veh_id) == veh_type:
                    presence[idx] = 1
                    break
        return presence

    def get_gps_status(self):
        """Simulates GPS data, return list od vehicles witch could be connected to the city network (potentially gps e.g. emergency and city_buses)
        witch left the simulation or rode through the junction (left the edges)"""
        conn = self.tc.getConnection(self.label) if self.gui else self.tc

        passed_priority = []

        exit_edges = ["E3","E4","E5","E6"]

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

    def get_pedestrian_presence(self):
        """Simulation of pedestrians buttons on crosswalks, return if they are any pedestrians on the edges"""
        ped_edges = ["-E4", "E3", "-E5", "-E6"]
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

    def get_emission_metrics(self):
        """Gets data about CO2 emissions and fuel consumption on intakes of the junction"""
        total_fuel = 0
        total_co2 = 0
        intake_edges = ["-E6", "E3", "-E4", "-E5"]

        conn = self.tc.getConnection(self.label) if self.gui else self.tc #dodane

        for edge_id in intake_edges:
            total_fuel += conn.edge.getFuelConsumption(edge_id)
            total_co2 += conn.edge.getCO2Emission(edge_id)

        return total_fuel, total_co2