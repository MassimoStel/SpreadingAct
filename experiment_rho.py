import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from SpreadPy.Models.sandpile import SandpileSpreading
import os

print("Caricamento rete per Auto-organizzazione rho(t)...")
df = pd.read_csv("data/SWOW-EN18/strength.SWOW-EN.R123.20180827.csv", sep='\t')
G_full = nx.from_pandas_edgelist(df, 'cue', 'response')

largest_cc = max(nx.connected_components(G_full), key=len)
G_core = G_full.subgraph(largest_cc).copy()

np.random.seed(42)

start_node = list(G_core.nodes())[0]
edges = list(nx.bfs_edges(G_core, start_node, depth_limit=3))
nodes = [start_node] + [v for u, v in edges]
nodes = nodes[:1000]   # stesso sottografo della transizione di fase (confronto rho_c)
G = G_core.subgraph(nodes).copy()
G.remove_edges_from(nx.selfloop_edges(G))

sink_node = max(G.degree, key=lambda x: x[1])[0]
sp = SandpileSpreading(G, sink=sink_node, f=0, strict=True)

rhos = []
iters = []

print("Simulazione in corso...")
for i in range(25000):
    res = sp.iteration()
    if res['diverged']: break
    rhos.append(res['total_grains'] / G.number_of_nodes())
    iters.append(i)

rhos = np.array(rhos)
rho_c = np.mean(rhos[len(rhos)//2:]) # stima della densità stazionaria (seconda metà)
print(f"rho_c stazionario stimato: {rho_c:.2f}")

plt.figure(figsize=(8,6))
plt.plot(iters, rhos, color='#9b59b6', label='Densità $\\rho(t)$')
plt.axhline(rho_c, color='k', linestyle='--', label=f'$\\rho_{{stat}} \\approx {rho_c:.2f}$ ($\\neq \\rho_c$)')
plt.title("Auto-organizzazione: convergenza a uno stato stazionario", fontsize=14)
plt.xlabel("Tempo (iterazioni)", fontsize=12)
plt.ylabel("Densità $\\rho$ (grani / nodi)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/soc_rho_vs_time.png", dpi=300)
print("Grafico salvato in figures/soc_rho_vs_time.png")
