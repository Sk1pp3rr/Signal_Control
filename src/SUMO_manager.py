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
        is_ambulance=False #ustawiamy flagi
        is_ambulance_stuck=False
        #logika sprawdzenia czy mamy na mapie ambulans
        vehicle_id=traci.vehicle.getIDList() #pobieramy cala liste samochodow ktore znajduja sie na mapie
        for i in vehicle_id:
            if traci.vehicle.getTypeID(i) == "ambulance": #sprawdzamy czy sposrod aut mamy ambulans
                is_ambulance=True #jesli tak to true

                # ponizej logika sprawdzenia czy ambulans stoi
                speed=traci.vehicle.getSpeed(i) #pobieramy predkosc
                if speed<0.5: #tutaj mozna sprawdzic czy taka predkosc bedzie git (najwyzej sie zmieni)
                    is_ambulance_stuck=True

                break #jesli znalezlismy to stop

        return is_ambulance, is_ambulance_stuck