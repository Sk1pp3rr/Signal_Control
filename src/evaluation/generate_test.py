import os
import random
from pathlib import Path


def generate_test_baseline():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_path = project_root / "maps" / "krzyzak" / "test_routes.rou.xml"

    routes_definitions = {
        # move through all traffic
        "route_WE": "E3 E5 E12 E13",
        "route_EW": "-E13 -E12 -E5 -E3",

        # --- Vertical J6 east ---
        "route_NS_J6": "-E4 E6",
        "route_SN_J6": "-E6 E4",

        # --- Vertical J8 middle ---
        "route_NS_J8": "-E10 E11",
        "route_SN_J8": "-E11 E10",

        # --- Vertical J15 west ---
        "route_NS_J15": "-E14 E15",
        "route_SN_J15": "-E15 E14",

        "route_N_to_E_J6": "-E4 E5 E12 E13",  #Enter from north to J6 and move west till the end
        "route_S_to_W_J8": "-E11 -E5 -E3",  #Enter from south to J8 and move west till the end
        "route_E_to_N_J15": "-E13 E14"  #J15 Enter from east and turn north on J15
    }

    ped_routes = [
        # Ped J6
        ("-E4", "E4"), ("E4", "-E4"), ("-E6", "E6"), ("E6", "-E6"),
        # Ped J8
        ("-E10", "E10"), ("E10", "-E10"), ("-E11", "E11"), ("E11", "-E11"),
        # Ped J15
        ("-E14", "E14"), ("E14", "-E14"), ("-E15", "E15"), ("E15", "-E15"),
        # Ped on edges
        ("E3", "-E3"), ("-E13", "E13")
    ]

    #Param of traffic
    max_steps = 86400
    # Probability 0.2 = light, 0.5 = medium, 0.8 = jam
    #car_probability = 0.2
    ped_probability = 0.05

    print(f"--- Gen testing baseline ---")

    with open(output_path, "w", encoding="UTF-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')

        f.write('    <vType id="ped_default" vClass="pedestrian" width="0.5" length="0.5" minGap="0.2"/>\n\n')

        for r_id, edges in routes_definitions.items():
            f.write(f'    <route id="{r_id}" edges="{edges}"/>\n')

        f.write('\n')

        #vehicle gen
        veh_count = 0
        ped_count = 0

        for step in range(max_steps):
            hour = (step // 3600) % 24

            if hour < 5:
                phase = 0  # Night
                car_probability = 0.005
            elif hour < 7:
                phase = 1  # Early morning
                car_probability = 0.05
            elif hour < 9:
                phase = 2  # Morning peak
                car_probability = 0.2
            elif hour < 15:
                phase = 3  # Day
                car_probability = 0.08
            elif hour < 18:
                phase = 4  # Afternoon peak
                car_probability = 0.18
            else:
                phase = 5  # Evening
                car_probability = 0.04

            # generator of cars
            if random.random() < car_probability:
                chosen_route = random.choice(list(routes_definitions.keys()))
                f.write(
                    f'    <vehicle id="test_veh_{veh_count}" type="urban_cars" depart="{step}.00" route="{chosen_route}"/>\n')
                veh_count += 1

                # Trucks early morning
                if phase == 1 and random.random() < 0.03:
                    f.write(
                        f'    <vehicle id="test_truck_{veh_count}" type="heavy_truck" depart="{step}.00" route="{chosen_route}"/>\n')
                    veh_count += 1

            # Pedestrain generation
            if random.random() < ped_probability:
                start_edge, end_edge = random.choice(ped_routes)
                f.write(f'    <person id="test_ped_{ped_count}" depart="{step}.00" type="ped_default">\n')
                f.write(f'        <walk from="{start_edge}" to="{end_edge}"/>\n')
                f.write(f'    </person>\n')
                ped_count += 1

        f.write('</routes>\n')

    print(f">>> SUCCESS: Generated {veh_count} vehicles and {ped_count} pedestrians to {output_path}")


if __name__ == "__main__":
    generate_test_baseline()