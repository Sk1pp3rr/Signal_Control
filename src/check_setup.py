import os
import sys
if 'SUMO_HOME' in os.environ:
    print(f"SUMO_HOME znalezione: {os.environ['SUMO_HOME']}")
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci
    print("Biblioteka TraCI załadowana pomyślnie!")
else:
    print("Błąd: Nie znaleziono zmiennej SUMO_HOME!")