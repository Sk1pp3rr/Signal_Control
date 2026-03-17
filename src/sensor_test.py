import os
import sys

# Pobieramy ścieżkę do folderu, w którym znajduje się ten skrypt (src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Budujemy ścieżkę: wyjdź wyżej (..), wejdź do maps -> krzyzak -> plik
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "maps", "krzyzak", "krzyzak.sumocfg"))

print(f"Szukam pliku w: {CONFIG_PATH}")

if not os.path.exists(CONFIG_PATH):
    print("❌ Wciąż nie widzę pliku! Sprawdź czy foldery się zgadzają.")
    # Tutaj możesz wypisać co widzi Python:
    print(f"Zawartość folderu nadrzędnego: {os.listdir(os.path.join(BASE_DIR, '..'))}")
    sys.exit()

import traci

DETECTORS = ["e2_0", "e2_1", "e2_2", "e2_3"]


def run_test():
    # Używamy r"..." dla ścieżek na Windowsie!
    sumo_cmd = ["sumo-gui", "-c", CONFIG_PATH, "--start"]

    try:
        traci.start(sumo_cmd)
        print("✅ SUMO uruchomione poprawnie!")
    except Exception as e:
        print(f"❌ Nie udało się uruchomić SUMO: {e}")
        return

    print("--- START TESTU DETEKTORÓW ---")

    step = 0
    while step < 1000:
        traci.simulationStep()

        for det_id in DETECTORS:
            try:
                jam_length = traci.lanearea.getJamLengthVehicle(det_id)
                veh_count = traci.lanearea.getLastStepVehicleNumber(det_id)

                if veh_count > 0:
                    print(f"Krok {step} | Detektor: {det_id} | Auta: {veh_count} | Korek: {jam_length}")
            except traci.exceptions.TraCIException:
                # To się stanie, jeśli ID detektora w Pythonie nie zgadza się z tym w .add.xml
                print(f"⚠️ Nie znaleziono detektora o ID: {det_id}")

        step += 1
        if traci.simulation.getMinExpectedNumber() <= 0:
            print("Wszystkie auta opuściły symulację.")
            break

    traci.close()
    print("--- KONIEC TESTU ---")


if __name__ == "__main__":
    run_test()