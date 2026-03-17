import traci
import time
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURACJA ---
SUMO_CMD = ["sumo-gui", "-c", "test.sumocfg", "--start"]
TLS_ID = "J25"  # ID skrzyżowania do sterowania
EDGES_NS = ["E13", "-E15"]  # ID ulic pionowych
EDGES_EW = ["E12", "-E14"]  # ID ulic poziomych
THRESHOLD = 10  # Zmień światło, gdy stoi tu więcej niż 10 aut


def get_queue_length(edges):
    """Liczy ile aut stoi (prędkość < 0.1 m/s) na podanych ulicach"""
    return sum([traci.edge.getLastStepHaltingNumber(e) for e in edges])


import random


def change_sensor(sensor_id, state):
    traci.trafficlight.setSensor(sensor_id, bool(state))

def sensor_malfunction():
    sensors = []
    duration = 0
    max_duration = random.random()
    if(random.Random % 100001 == 0):
        TLS_ID = random.choice(sensors)
        change_sensor(TLS_ID, False)
        duration = 0
    duration += 1
    if duration >= max_duration:
        change_sensor(TLS_ID, True)

def stworz_wypadek():
    ulice_wlotowe = ["E13", "E12", "-E14", "-E15"]
    ulica = random.choice(ulice_wlotowe)
    auta = traci.edge.getLastStepVehicleIDs(ulica)
    if auta:
        traci.vehicle.setStop(auta[0], ulica, pos=20, duration=1000)
        print(f"WYPADEK na {ulica}!")


def wykresKolejek(steps, queue_ns_data, queue_ew_data):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, queue_ns_data, label='Kolejka Północ-Południe', color='blue')
    plt.plot(steps, queue_ew_data, label='Kolejka Wschód-Zachód', color='red')
    plt.title('Długość kolejki w czasie')
    plt.xlabel('Sekundy symulacji')
    plt.ylabel('Liczba stojących aut')
    plt.legend()
    plt.grid(True)
    plt.savefig('wykres_kolejek.png')
    plt.show()


def wykresOczeikiwania(steps, waiting_time_data):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, waiting_time_data, color='green')
    plt.title('Łączny czas oczekiwania na skrzyżowaniu')
    plt.xlabel('Sekundy symulacji')
    plt.ylabel('Suma czasu oczekiwania (s)')
    plt.grid(True)
    plt.savefig('wykres_czekania.png')
    plt.show()


def run_smart_simulation():
    traci.start(SUMO_CMD)

    step = 0
    waiting_times_history = []
    steps = []
    queue_ns_data = []
    queue_ew_data = []
    waiting_time_data = []
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        q_ns = get_queue_length(EDGES_NS)
        q_ew = get_queue_length(EDGES_EW)
        total_wait = sum([traci.edge.getWaitingTime(e) for e in EDGES_NS + EDGES_EW])

        steps.append(step)
        queue_ns_data.append(q_ns)
        queue_ew_data.append(q_ew)
        waiting_time_data.append(total_wait)

        current_waiting_time = sum([traci.edge.getWaitingTime(e) for e in EDGES_NS + EDGES_EW])
        waiting_times_history.append(current_waiting_time)

        # Pobieramy aktualną fazę (0 i 2 to zazwyczaj zielone dla różnych kierunków)
        current_phase = traci.trafficlight.getPhase(TLS_ID)

        queue_ns = get_queue_length(EDGES_NS)
        queue_ew = get_queue_length(EDGES_EW)

        random_event = (step % 5000 == 0)  # Co 300 kroków symulacji wprowadzamy losowe zdarzenie
        if random_event:
            stworz_wypadek()
        # Logika sterowania:
        # Jeśli świeci się zielone dla Północ-Południe (faza 0), 
        # a na Wschód-Zachód stoi duża kolejka -> przełączamy.
        if current_phase == 0 and queue_ew > queue_ns + THRESHOLD:
            #print(f"Korek na E-W ({queue_ew} aut)! Zmieniam na zielone dla nich.")
            traci.trafficlight.setPhase(TLS_ID, 2)  # Przełącz na fazę 2 (zielone E-W)

        elif current_phase == 2 and queue_ns > queue_ew + THRESHOLD:
            #print(f"Korek na N-S ({queue_ns} aut)! Zmieniam na zielone dla nich.")
            traci.trafficlight.setPhase(TLS_ID, 0)  # Przełącz na fazę 0 (zielone N-S)

        step += 1
    avg_waiting = sum(waiting_times_history) / len(waiting_times_history)
    max_waiting = max(waiting_times_history)

    print("\n--- RAPORT EFEKTYWNOŚCI ---")
    print(f"Średni czas oczekiwania w całej symulacji: {avg_waiting:.2f} s")
    print(f"Maksymalny zator (w sekundach stania): {max_waiting:.2f} s")
    print(f"Całkowity czas symulacji: {step} kroków")
    wykresKolejek(steps, queue_ns_data, queue_ew_data)
    wykresOczeikiwania(steps, waiting_time_data)
    traci.close()


if __name__ == "__main__":
    run_smart_simulation()
