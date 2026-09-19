import traci
import random
from core.SUMO_manager import SumoManager
from core.random_events_func import eventManager


def run_fixed_baseline(config_path, max_sim_time=3600, seed=42):
    print("--- START BASELINE: FIXED TIME ---")
    random.seed(seed)

    sumo = SumoManager(config_path, gui=False, rank=10)
    sumo.start_sim()
    events = eventManager(sumo)
    conn = traci.getConnection(sumo.label)

    metrics_log = {
        'queue_history': [],
        'halting_history': [],
        'priority_wait_dict': {}
    }

    macro_step = 0
    last_macro_time = 0

    try:
        while conn.simulation.getTime() < max_sim_time:
            current_time = conn.simulation.getTime()

            if current_time - last_macro_time >= 5.0:
                macro_step += 1

                events.emergnecy_vechicle_deployment(probability=0.01)
                events.scheduled_bus_deployment(macro_step, "route_NS", stops=["busStop_J6_South"], line_name="101_A",
                                                interval_steps=150)
                events.scheduled_bus_deployment(macro_step, route_id="route_SN", stops=["busStop_J6_North"],
                                                line_name="101_B", interval_steps=180)
                events.scheduled_bus_deployment(macro_step, route_id="route_WE", stops=["busStop_J6_East"],
                                                line_name="102", interval_steps=200)
                events.scheduled_bus_deployment(macro_step, route_id="route_EW", stops=["busStop_J6_West"],
                                                line_name="103", interval_steps=200)
                events.detector_malfunction()

                j_metrics = sumo.get_junction_metrics()
                metrics_log['queue_history'].append(j_metrics['max_jam_length'])
                metrics_log['halting_history'].append(j_metrics['total_halting'])

                passed = sumo.get_gps_status()
                for v in passed:
                    if v['type'] in ["ambulance", "city_bus"]:
                        veh_id = v['id']
                        current_wait = v.get('wait', 0)
                        metrics_log['priority_wait_dict'][veh_id] = max(
                            metrics_log['priority_wait_dict'].get(veh_id, 0), current_wait)

                last_macro_time = current_time

            conn.simulationStep()

            if conn.simulation.getMinExpectedNumber() <= 0:
                break

    finally:
        sumo.close_sim()

    print("--- END FIXED TIME ---")
    avg_queue = sum(metrics_log['queue_history']) / max(1, len(metrics_log['queue_history']))
    avg_halting = sum(metrics_log['halting_history']) / max(1, len(metrics_log['halting_history']))
    total_priority_wait = sum(metrics_log['priority_wait_dict'].values())

    final_metrics = {
        'queue_length': avg_queue,
        'total_halting': avg_halting,
        'priority_wait_time': total_priority_wait
    }

    return final_metrics

if __name__ == "__main__":
    import os
    config = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "..", "maps", "krzyzak", "krzyzak.sumocfg"))
    print(run_fixed_baseline(config))