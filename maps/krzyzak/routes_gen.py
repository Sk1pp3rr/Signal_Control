import os
import random
import sumolib

script_dir = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(script_dir, 'czysta_morska_final.net.xml')
OUTPUT_FILE = os.path.join(script_dir, 'routes_logical.rou.xml')

# Wpisz swoje ID krawędzi wlotowych i wylotowych
intake_edges = ["-286967103", "169121911#1", "-27398735","-177211203#1","Kcynska_3in","27398644#1","53533699","27399957#0"]
exit_edges = ["136687277#2", "180623102#1", "Owsiana_1out","27398735","177211203#0","-27399957#5","Kcynska_2out","Kcynska_4out","-53533699"]


def generate_logical_routes():
    print("Ładowanie sieci...")
    net = sumolib.net.readNet(NET_FILE)

    # 1. Wybierz krawędzie wlotowe i wylotowe (tylko z zewnątrz mapy)
    # Zmienisz to na swoje konkretne ID, jeśli nazwy są inne
    inlets = intake_edges
    outlets = exit_edges

    print(f"Znaleziono {len(inlets)} wlotów i {len(outlets)} wylotów.")

    with open(OUTPUT_FILE, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        f.write('    <vType id="car" accel="2.6" decel="4.5" length="5.0" maxspeed="70.0"/>\n')

        route_count = 0
        for start_id in inlets:
            for end_id in outlets:
                start_edge = net.getEdge(start_id)
                end_edge = net.getEdge(end_id)

                if start_edge and end_edge:
                    # Funkcja zwraca (lista_krawedzi, koszt) lub samą listę
                    result = net.getShortestPath(start_edge, end_edge)

                    # Logika bezpiecznego pobrania krawędzi:
                    path = result[0] if isinstance(result, tuple) else result

                    if path:
                        # Teraz e jest już pojedynczą krawędzią (Edge)
                        edge_ids = [e.getID() for e in path]

                        f.write(f'    <route id="route_{route_count}" edges="{" ".join(edge_ids)}"/>\n')
                        route_count += 1
                else:
                    print(f"Uwaga: Nie znaleziono krawędzi {start_id} lub {end_id} w sieci!")


        f.write('</routes>\n')

    print(f"Gotowe! Wygenerowano {route_count} logicznych tras w: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_logical_routes()