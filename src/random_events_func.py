import traci
import random

from setuptools import Extension
from sympy.codegen.fnodes import Extent


class eventManager:
    def __init__(self, sumo_env):
        self.env = sumo_env
        self.available_routes = []


    """Added two functions for further training like if we would like to train this agent on the base of the whole day with it's own phases
    for further reality"""
    def get_time(self,current_step):
        hour_step = 720
        day_time = current_step % (24 * hour_step)

        if day_time < 5 * hour_step: return 0  # Night (00:00 - 05:00)
        if day_time < 7 * hour_step: return 1  # Early morning (05:00 - 07:00)
        if day_time < 9 * hour_step: return 2  # Morning peak (07:00 - 09:00)
        if day_time < 15 * hour_step: return 3  # Day (09:00 - 15:00)
        if day_time < 18 * hour_step: return 4  # Afternoon peak (15:00 - 18:00)
        return 5  # Evening (18:00 - 00:00)

    def spawn_dynamic_traffic(self, current_step, route_id):
        phase = self.get_time_phase(current_step)

        # Probablity of car spawn dependent on hour
        probs = {
            0: 0.005,
            1: 0.05,
            2: 0.2,
            3: 0.08,
            4: 0.18,
            5: 0.04
        }

        if random.random() < probs[phase]:
            veh_id = f"veh_{current_step}"
            # Using distribution from vTypeDistribution
            traci.vehicle.add(veh_id, route_id, typeID="urban_cars")

        # Spawning bigger trucks only in the early morning
        if phase == 1 and random.random() < 0.03:
            traci.vehicle.add(f"truck_{current_step}", route_id, typeID="heavy_truck")




    #makes list of all avaliable routes defined in simulation
    def find_rotes(self):
        all_routes = traci.route.getIDList()
        self.available_routes = [route for route in all_routes if not route.startswith('!')]
        return

    def collision(self):
        if not self.avaliable_routes:
            self.find_rotes()
        # Random choice of lane
        target_lane = random.choice(self.available_routes)
        traci.lane.setDisallowed(target_lane, ["passenger", "bus", "truck"])
        return

    def detector_malfunction(self):
        #TODO: implementation of detector error, maybe dependent on detector type. If so implement different function and call them here
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

    #I know it is not random if it is scheduled, but we ball
    def scheduled_bus_deployment(self, current_step, route_id,stops = None, line_name="101", interval_steps=100):
        """
        It leaves bus in fixed time stamps on track, and gives them stops
        interval_steps=100 is aprox. 8-9 with step duration 5s. We have it 5 or 8, 8 when it changes phases
        """
        # Check if there is time to deploy the bus
        if current_step % interval_steps == 0 and current_step > 0:
            veh_id = f"bus_{line_name}_{current_step}"
            try:
                # Using vType form XML file
                traci.vehicle.add(veh_id, route_id, typeID="city_bus")
                traci.vehicle.setLine(veh_id, line_name)
                # If there are any problem with colour: traci.vehicle.setColor(veh_id, (255, 255, 0))

                #Connecting buses with stops
                if stops:
                    for stop_id in stops:
                        traci.vehicle.setBusStop(veh_id, stop_id, duration=20)

            except traci.TraCIException as e:
                # If there are any issues we don't want to destroy anything
                pass