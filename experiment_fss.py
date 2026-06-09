import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from SpreadPy.Models.sandpile import SandpileSpreading
import os

print("Caricamento rete SWOW per Finite-Size Scaling...")
df = pd.read_csv("data/SWOW-EN18/strength.SWOW-EN.R123.20180827.csv", sep='\t')
G_full = nx.from_pandas_edgelist(df, 'cue', 'response')

largest_cc = max(nx.connected_components(G_full), key=len)
G_core = G_full.subgraph(largest_cc).copy()

sizes = [250, 500, 1000, 2500]
results = {}

start_node = list(G_core.nodes())[0]

for N in sizes:
    print(f"\n--- Simulazione per N={N} ---")
    edges = list(nx.bfs_edges(G_core, start_node, depth_limit=4))
    nodes = [start_node] + [v for u, v in edges]
    if len(nodes) > N:
        nodes = nodes[:N]
    
    G = G_core.subgraph(nodes).copy()
    print(f"Grafo estratto: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")
    
    sink_node = max(G.degree, key=lambda x: x[1])[0]
    
    sp = SandpileSpreading(G, sink=sink_node, f=0, strict=True)
    avalanches = []
    
    print("Burn-in...")
    for i in range(sp.sum_degrees):
        res = sp.iteration()
        if res['diverged']: break
        
    print("Raccolta valanghe...")
    for i in range(20000):
        res = sp.iteration()
        if res['diverged']: break
        if res['has_avalanche']:
            avalanches.append(res['avalanche_size'])
            
    avalanches = np.array(avalanches)
    results[N] = avalanches
    print(f"Valanghe per N={N}: {len(avalanches)}")

os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(8,6))

colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
import powerlaw

for i, N in enumerate(sizes):
    avs = results[N]
    if len(avs) == 0: continue
    powerlaw.plot_pdf(avs, color=colors[i], label=f'N={N}', linear_bins=False, linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.title("Finite-Size Scaling: Cutoff Dipendente dalla Dimensione", fontsize=14, pad=15)
plt.xlabel("Dimensione della Valanga $S$", fontsize=12)
plt.ylabel("Densità di Probabilità $P(S)$", fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig("figures/finite_size_scaling.png", dpi=300)
print("Grafico salvato in figures/finite_size_scaling.png")
