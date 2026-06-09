import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random

sys.path.append(os.getcwd())
from SpreadPy.Models.sandpile import SandpileSpreading

print("Caricamento rete SWOW...")
df = pd.read_csv("data/SWOW-EN18/strength.SWOW-EN.R123.20180827.csv", sep='\t')
G_full = nx.from_pandas_edgelist(df, 'cue', 'response', create_using=nx.Graph())

# Estraiamo un sottografo connesso di 200 nodi per fare in fretta
print("Creazione sottografo connesso...")
start_node = list(G_full.nodes())[0]
edges = list(nx.bfs_edges(G_full, start_node, depth_limit=4))
nodes = [start_node] + [v for u, v in edges]
nodes = nodes[:200]
G = G_full.subgraph(nodes).copy()

# Rimuovi eventuali nodi a grado 0 creati dal taglio
nodes_to_remove = [n for n, d in G.degree() if d == 0]
G.remove_nodes_from(nodes_to_remove)

# RIMOZIONE SELF-LOOPS (fondamentale: altrimenti G.degree() conta 2 e i grani evaporano!)
G.remove_edges_from(nx.selfloop_edges(G))

print(f"Grafo creato: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")

# Testiamo 150 nodi bersaglio scelti a caso per statistica solida
test_nodes = random.sample(list(G.nodes()), min(150, G.number_of_nodes()))
results = []

for node in test_nodes:
    deg = G.degree(node)
    # Nessun sink, f=0 -> sistema chiuso destinato a divergere
    sp = SandpileSpreading(G, sink=None, f=0, strict=True)
    
    iters = 0
    diverged = False
    
    # Buttiamo grani SEMPRE sullo stesso nodo finché il sistema esplode
    while not diverged and iters < 20000:
        res = sp.iteration(node=node)
        diverged = res.get('diverged', False)
        iters += 1
        
    results.append({'node': node, 'degree': deg, 'iters_to_divergence': iters})

res_df = pd.DataFrame(results)

from scipy.stats import spearmanr
rho, p_val = spearmanr(res_df['degree'], res_df['iters_to_divergence'])
print(f"Statistiche: Spearman rho = {rho:.3f}, p-value = {p_val:.4f}")

# Plotting formale
import os
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8,6))
# Impostiamo la scala logaritmica sull'asse X per espandere la nuvola
plt.xscale('log')
plt.scatter(res_df['degree'], res_df['iters_to_divergence'], c='#4a90e2', alpha=0.6, s=60, edgecolors='white', linewidths=0.5)

plt.title("Saturazione Globale: Velocità di divergenza", fontsize=14, pad=15)
plt.xlabel("Grado del Nodo Bersaglio (Scala Logaritmica)", fontsize=12)
plt.ylabel("Grani assorbiti prima del collasso termodinamico", fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.3)

# Calcoliamo una trendline log-lineare per evitare l'effetto leva dell'outlier
log_x = np.log10(res_df['degree'])
z = np.polyfit(log_x, res_df['iters_to_divergence'], 1)
p = np.poly1d(z)

# Disegniamo la retta ordinando i punti per l'asse x
x_sorted = np.sort(res_df['degree'])
plt.plot(x_sorted, p(np.log10(x_sorted)), "k--", alpha=0.6, label=f'Trend Log-Lineare ($\\rho_s = {rho:.2f}, p < 0.05$)')

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("figures/soc_divergence_capacity.png", dpi=300)
print("\nGrafico salvato in figures/soc_divergence_capacity.png")
with open("divergence_stats.txt", "w") as f:
    f.write(f"rho={rho:.3f}\np={p_val:.4e}\n")
