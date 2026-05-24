import random

in_edges = ["-286967103","27399957#0","53533699","27398644#1","Kcynska_3in","-177211203#1","169121911#1","-27398735"]
out_edges = ["Kcynska_2out","-53533699","Kcynska_4out","-27399957#5","136687277#2","180623102#1","Owsiana_1out","27398735","177211203#0"]

with open("base_routes_fixed.rou.xml", "w") as f:
    f.write('<routes>\n  <vType id="car" accel="2.6" decel="4.5" length="5"/>\n\n')

    for i, start in enumerate(in_edges):
        for j, end in enumerate(out_edges):
            if i != j:  # Nie jedziemy z powrotem tą samą drogą
                r_id = f"route_{i}_{j}"
                f.write(f'  <route id="{r_id}" edges="{start} {end}"/>\n')
                f.write(
                    f'  <flow id="flow_{r_id}" type="car" begin="0" end="86400" period="{random.randint(15, 60)}" route="{r_id}"/>\n\n')
    f.write('</routes>')