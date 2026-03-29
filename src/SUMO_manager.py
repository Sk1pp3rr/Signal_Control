import traci

class SumoManager:
    def __init__(self, config_path, gui=False):
        self.config_path = config_path
        self.sumo_cmd = ["sumo-gui" if gui else "sumo", "-c", self.config_path] #Manager will automatically work dependably of weather gui is enabled
        self.DETECTORS = ["e2_0", "e2_1", "e2_2", "e2_3"] #table of detectors in "krzyzak"

    def start_sim(self):
        """Start the SUMO simulator"""
        traci.start(self.sumo_cmd)

    def close_sim(self):
        """Close the SUMO simulator"""
        if traci.isLoaded():
            traci.close()

    def set_traffic_light_phase(self,phase,junction_id):
        """Set the traffic light phase on the specified junction_id"""
        traci.trafficlight.setPhase(phase, junction_id)

    def get_dettector_data(self):
        """Get the detection detector data"""
        data = []
        for detector in self.DETECTORS:
            count = traci.lanearea.getLastStepVehicleNumber(detector) #get the data from individual detector
            data.append(count) # add them to our vector
        return data # e.g. [3,2,0,5]


