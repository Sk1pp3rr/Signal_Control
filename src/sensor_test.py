import os
import sys

# Downloading path where file is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# path from src to sumo config
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "maps", "krzyzak", "krzyzak.sumocfg"))

print(f"Looking for file in: {CONFIG_PATH}")

if not os.path.exists(CONFIG_PATH):
    print("file not found")
    # Python view:
    print(f"Upstream folder: {os.listdir(os.path.join(BASE_DIR, '..'))}")
    sys.exit()

import traci

DETECTORS = ["e2_0", "e2_1", "e2_2", "e2_3"]


def run_test():
    sumo_cmd = ["sumo-gui", "-c", CONFIG_PATH, "--start"]

    try:
        traci.start(sumo_cmd)
        print("SUMO boot success")
    except Exception as e:
        print(f"SUMO boot failed: {e}")
        return

    print("--- DETECTORS TEST START ---")

    step = 0
    while step < 1000:
        traci.simulationStep()

        for det_id in DETECTORS:
            try:
                jam_length = traci.lanearea.getJamLengthVehicle(det_id)
                veh_count = traci.lanearea.getLastStepVehicleNumber(det_id)

                if veh_count > 0:
                    print(f"Step {step} | Detector: {det_id} | Cars: {veh_count} | Jam: {jam_length}")
            except traci.exceptions.TraCIException:
                # ID of detector in python do not match with this in .add.xml
                print(f"️ failed to find detector: {det_id}")

        step += 1
        if traci.simulation.getMinExpectedNumber() <= 0:
            print("All cars left simulation.")
            break

    traci.close()
    print("--- END OF TEST ---")


if __name__ == "__main__":
    run_test()