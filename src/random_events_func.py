import traci
import random

from setuptools import Extension
from sympy.codegen.fnodes import Extent


class eventManager:
    def __init__(self, sumo_env):
        self.env = sumo_env
        self.available_routes = []
        self.detector_status = [1,1,1,1] #is detector working 1->yes , 0->no la policiaaa :(

    #makes list of all avaliable routes defined in simulation
    def find_rotes(self):
        all_routes = traci.route.getIDList()
        self.available_routes = [route for route in all_routes if not route.startswith('!')]
        return

    def collision(self):
        #TODO: implementation of trafic stop, i think it is valid
        if not self.avaliable_routes:
            self.find_rotes()
        # Random choice of lane
        target_lane = random.choice(self.available_routes)
        traci.lane.setDisallowed(target_lane, ["passenger", "bus", "truck"])
        return

    def detector_malfunction(self, probability=0.002):
        #TODO: implementation of detector error, maybe dependent on detector type. If so implement different function and call them here
        detectors = self.env.DETECTORS
        prawd=random.random() #chose  number between (0,1)
        if prawd < probability:
            random_idx = random.randint(0, len(detectors)-1)
            self.detector_status[random_idx]=0
        return

    def emergnecy_vechicle_deployment(self, probability = 0.001):
        if random.random() < probability:
            if not self.available_routes:
                self.find_rotes()
            # Random choice of lane
            route_id = random.choice(self.available_routes)
            veh_id = f"emergency_{traci.simulation.getTime()}"

            try:
                traci.vehicle.add(veh_id, route_id, typeID="ambulance")
                traci.vehicle.setColor(veh_id, (255, 0, 0, 255))
                print(f"Ambulance on route id: {route_id}")
            except Extension as e:
                print(f"Error: {e}") #debugging
        return
