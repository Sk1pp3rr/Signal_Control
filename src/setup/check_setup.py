import os
import sys
if 'SUMO_HOME' in os.environ:
    print(f"SUMO_HOME znalezione: {os.environ['SUMO_HOME']}")
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci
    print("Library loaded successfully")
else:
    print("Error: Did not found env variable SUMO_HOME!")