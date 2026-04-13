from asyncio import wait

import traci
from sympy import true
from torch.fx.experimental.proxy_tensor import track_tensor


class SumoManager:
    def __init__(self, config_path, gui=False):
        self.config_path = config_path
        self.sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", self.config_path, "--start"] #Manager will automatically work dependably of weather gui is enabled
        self.DETECTORS = ["e2_0", "e2_1", "e2_2", "e2_3"] #table of detectors in "krzyzak"

    def start_sim(self):
        """Start the SUMO simulator"""
        traci.start(self.sumo_cmd)

    def close_sim(self):
        """Close the SUMO simulator"""
        if traci.isLoaded():
            traci.close()

    def set_traffic_light_phase(self,junction_id, phase):
        """Set the traffic light phase on the specified junction_id"""
        traci.trafficlight.setPhase(junction_id, phase)

    def get_detector_data(self):
        """Get the detection detector data"""
        data = []
        for detector in self.DETECTORS:
            count = traci.lanearea.getLastStepVehicleNumber(detector) #get the data from individual detector
            data.append(count) # add them to our vector
        return data # e.g. [3,2,0,5]

    def get_waiting_time_data(self):
        """Get the waiting time detector data"""
        total_waiting_time = 0
        for detector in self.DETECTORS:
            # Get total weiting time from each generator
            total_waiting_time += traci.lanearea.getWaitingTime(detector)
        return total_waiting_time

    def get_junction_metrics(self):
        """Download raw data from SUMO junction"""
        metrics = {
            'total_halting': 0,
            'max_jam_length': 0,
            'occupancy': 0
        }
        for detector in self.DETECTORS:

            metrics['total_halting'] += traci.lanearea.getLastStepHaltingNumber(detector)
            metrics['max_jam_length'] += traci.lanearea.getLastIntervalMaxJamLengthInMeters(detector)
            metrics['occupancy'] += traci.lanearea.getLastStepOccupancy(detector)
        return metrics

     #funkja sprawdza czy mamy ambulans na mapie lub czy utknela w korku
    def get_ambulance_metrics(self):

        vehicle_id=traci.vehicle.getIDList()
        ambulances=[] # were prepering list for ambulances
        for i in vehicle_id:
            if traci.vehicle.getTypeID(i) == "ambulance":
                is_ambulance=True
                is_ambulance_stuck = False
                ambulance_lane = None

                waiting_time=traci.vehicle.getWaitingTime(i) #we check how long ambulance are waiting
                if waiting_time > 5:
                    is_ambulance_stuck=True
                for detector in self.DETECTORS:
                    cars_in_detecor=traci.lanearea.getLastStepVehicleIDs(detector) #list of cars that are in range of detector
                    if i in cars_in_detecor:
                        ambulance_lane=detector
                wektor=[is_ambulance,is_ambulance_stuck, ambulance_lane]
                ambulances.append(wektor)


        return ambulances

    # This implementation is weird because I came to conclusion that not every ambulance will come from the "city of origin" where this agent could be working
    # sooo the agent can get the data both ways, not to neglect the existence of ambulances not integrated with the system.
    def get_veh_presence(self, veh_type):
        """Returns vector [0,0,0,0] with 1, in place where there is veh_type (GPS or Sensor)"""
        presence=[0,0,0,0]
        for idx, det_id in enumerate(self.DETECTORS):
            vehicle_on_det = traci.lanearea.getLastStepVehicleIDs(det_id)
            for veh_id in vehicle_on_det:
                if traci.vehicle.getTypeID(veh_id) == veh_type:
                    presence[idx] = 1
                    break
        return presence

    def get_gps_status(self):
        """Simulates GPS data, return list od vehicles witch could be connected to the city network (potentially gps e.g. emergency and city_buses)
        witch left the simulation or rode through the junction (left the edges)"""

        passed_priority = []

        exit_edges = ["E3","E4","E5","E6"]

        for edge_id in exit_edges:
            # get all the car's id on intake edges
            vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
            for v_id in vehicles:
                if traci.vehicle.getDistance(v_id) < 10.0:
                    v_type = traci.vehicle.getTypeID(v_id)
                    if v_type in ["ambulance","city_bus","police","fire_truck"]:
                        passed_priority.append({'type': v_type, 'wait': traci.vehicle.getWaitingTime(v_id)})
        return passed_priority