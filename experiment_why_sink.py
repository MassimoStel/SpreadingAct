"""
Genera figures/soc_why_sink.png: perche' serve il sink.
Sistema chiuso (sink=None, f=0): la massa cresce di 1 per iterazione e le valanghe
conservano i grani, quindi il sistema diverge SOLO quando la massa supera la somma
dei gradi (max_stable_mass). Usiamo un guard max_topplings alto in modo che la
divergenza rilevata coincida col tetto fisico, non con un falso allarme del guard.
Con un nodo-sink, invece, la massa si stabilizza ben sotto il tetto.
"""
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from SpreadPy.Models.sandpile import SandpileSpreading

np.random.seed(42)

print("Caricamento rete SWOW...")
df = pd.read_csv("data/SWOW-EN18/strength.SWOW-EN.R123.20180827.csv", sep='\t')
G_full = nx.from_pandas_edgelist(df, 'cue', 'response')
largest_cc = max(nx.connected_components(G_full), key=len)
G_core = G_full.subgraph(largest_cc).copy()

start_node = list(G_core.nodes())[0]
edges = list(nx.bfs_edges(G_core, start_node, depth_limit=3))
nodes = ([start_node] + [v for u, v in edges])[:400]
G = G_core.subgraph(nodes).copy()
G.remove_edges_from(nx.selfloop_edges(G))
print(f"Grafo: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")

# guard alto: lasciamo finire le valanghe grandi ma finite, cosi' la divergenza
# rilevata e' quella VERA (massa > somma gradi), non un falso positivo del guard
BIG = 2_000_000

# --- sistema chiuso: deve divergere al tetto ---
sp_closed = SandpileSpreading(G, sink=None, f=0, strict=True, max_topplings=BIG)
ceiling = sp_closed.max_stable_mass    # = somma dei gradi
print(f"Tetto (somma gradi) = {ceiling}")

mass_closed = []
div_iter = None
for i in range(ceiling + 50):
    res = sp_closed.iteration()
    if res['diverged']:
        div_iter = i
        div_mass = (mass_closed[-1] + 1) if mass_closed else 1
        print(f"Divergenza a iter={div_iter}, massa~={div_mass}")
        break
    mass_closed.append(res['total_grains'])

# --- con sink: massa stazionaria ben sotto il tetto ---
sink_node = max(G.degree, key=lambda x: x[1])[0]
sp_sink = SandpileSpreading(G, sink=sink_node, f=0, strict=True)
mass_sink = []
for i in range(ceiling + 50):
    res = sp_sink.iteration()
    mass_sink.append(res['total_grains'])

# --- figura ---
plt.figure(figsize=(8, 6))
plt.plot(range(len(mass_closed)), mass_closed, color='#e74c3c', linewidth=1.6,
         label='Sistema chiuso (sink=None, $f=0$)')
if div_iter is not None:
    plt.scatter([div_iter], [div_mass], color='#c0392b', marker='X', s=120, zorder=5,
                label='Divergenza: non si stabilizza più (fase attiva)')
plt.plot(range(len(mass_sink)), mass_sink, color='#2980b9', linewidth=1.2,
         label='Con nodo-sink (stato stazionario)')
plt.axhline(ceiling, color='k', linestyle=':', linewidth=1.2,
            label=f'Limite teorico massimo (somma gradi = {ceiling})')

plt.title("Perché serve il sink: il sistema chiuso smette di stabilizzarsi", fontsize=13)
plt.xlabel("Iterazione $t$ (grani aggiunti dall'esterno)", fontsize=12)
plt.ylabel("Massa totale (grani sulla rete)", fontsize=12)
plt.legend(loc='upper left', fontsize=9, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/soc_why_sink.png", dpi=200)
print("Figura salvata in figures/soc_why_sink.png")
