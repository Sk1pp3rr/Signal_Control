import os
import random
from pathlib import Path


def generate_test_baseline():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_path = project_root / "maps" / "krzyzak" / "test_routes.rou.xml"

    routes_definitions = {
        "route_NS": "-E4 -E3", "route_SN": "E3 E4",
        "route_WE": "-E6 E5", "route_EW": "-E5 E6",
        "route_N_S": "-E6 -E3", "route_N_E": "-E6 E5", "route_N_W": "-E6 E4",
        "route_S_N": "E3 E6", "route_S_E": "E3 E5", "route_S_W": "E3 E4",
        "route_E_N": "-E5 E6", "route_E_S": "-E5 -E3", "route_E_W": "-E5 E4",
        "route_W_N": "-E4 E6", "route_W_S": "-E4 -E3", "route_W_E": "-E4 E5"
    }

    ped_routes = [
        ("-E4", "-E3"), ("E3", "E4"),
        ("-E6", "E5"), ("-E5", "E6")
    ]

    #Param of traffic
    max_steps = 3600
    # Probability 0.2 = light, 0.5 = medium, 0.8 = jam
    car_probability = 0.2
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
            if random.random() < car_probability:
                # Random path
                chosen_route = random.choice(list(routes_definitions.keys()))

                f.write(
                    f'    <vehicle id="test_veh_{veh_count}" type="urban_cars" depart="{step}.00" route="{chosen_route}"/>\n')
                veh_count += 1

            if random.random() < ped_probability:
                start_edge, end_edge = random.choice(ped_routes)
                f.write(f'    <person id="test_ped_{ped_count}" depart="{step}.00" type="ped_default">\n')
                f.write(f'        <walk from="{start_edge}" to="{end_edge}"/>\n')
                f.write(f'    </person>\n')
                ped_count += 1

        f.write('</routes>\n')

    print(f">>> SUCCESS: Generated {veh_count} vehicles and {ped_count} pedestrians: {output_path}")


if __name__ == "__main__":
    generate_test_baseline()