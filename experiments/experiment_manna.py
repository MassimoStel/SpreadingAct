import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import powerlaw
import os

print("Caricamento rete SWOW...")
df = pd.read_csv("data/SWOW-EN18/strength.SWOW-EN.R123.20180827.csv", sep='\t')

# Creiamo un grafo diretto (asimmetrico) usando i pesi reali come probabilità
G_full = nx.from_pandas_edgelist(df, 'cue', 'response', edge_attr='R123.Strength', create_using=nx.DiGraph())

# Estraiamo un sottografo per mantenere i tempi rapidi
start_node = list(G_full.nodes())[0]
edges = list(nx.bfs_edges(G_full, start_node, depth_limit=3))
nodes = [start_node] + [v for u, v in edges]
nodes = nodes[:1000]
G = G_full.subgraph(nodes).copy()

print(f"Grafo Diretto creato: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")

# Precalcolo le probabilità di routing per il Modello di Manna
routing_probs = {}
for n in G.nodes():
    out_edges = list(G.out_edges(n, data=True))
    if not out_edges:
        routing_probs[n] = None # Nodo senza uscite (sink implicito)
    else:
        neighbors = [v for u, v, d in out_edges]
        weights = np.array([float(d.get('R123.Strength', 1.0)) for u, v, d in out_edges])
        if weights.sum() > 0:
            probs = weights / weights.sum()
        else:
            probs = np.ones(len(weights)) / len(weights)
        routing_probs[n] = (neighbors, probs)

# Aggiungo un sink esplicito
sink_node = max(G.out_degree, key=lambda x: x[1])[0]

status = {n: 0 for n in G.nodes()}
avalanches = []
max_iters = 50000

print("Esecuzione Modello di Manna stocastico...")
for i in range(max_iters):
    target = random.choice(list(G.nodes()))
    status[target] += 1
    
    unstable = [n for n in G.nodes() if status[n] >= 2 and n != sink_node]
    avalanche_size = 0
    loop_safeguard = 0
    
    while unstable and loop_safeguard < 10000:
        for v in unstable:
            # Manna: perde 2 grani alla volta
            status[v] -= 2
            avalanche_size += 2
            
            # Li distribuisce casualmente basandosi sui pesi (forza associativa)
            route_info = routing_probs[v]
            if route_info is not None:
                neighbors, probs = route_info
                chosen = np.random.choice(neighbors, size=2, p=probs)
                for u in chosen:
                    if u != sink_node:
                        status[u] += 1
                        
        unstable = [n for n in G.nodes() if status[n] >= 2 and n != sink_node]
        loop_safeguard += 1
        
    if avalanche_size > 0:
        avalanches.append(avalanche_size)
        
    if i > 0 and i % 10000 == 0:
        print(f"  Iterazione {i}, valanghe registrate: {len(avalanches)}")

avalanches = np.array(avalanches)
print(f"Valanghe totali: {len(avalanches)}")

# Fit e Plot
if len(avalanches) > 10:
    results = powerlaw.Fit(avalanches, discrete=True, xmin=2)
    alpha = results.power_law.alpha
    print(f"Fit completato: alpha = {alpha:.3f}")

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(8,6))
    
    powerlaw.plot_ccdf(avalanches, color='#e74c3c', linewidth=2, label=f'Valanghe (Manna Stocastico)')
    results.power_law.plot_ccdf(color='k', linestyle='--', label=f'Fit Legge di Potenza ($\\alpha={alpha:.2f}$)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Sandpile Stocastico (Grafo Diretto Asimmetrico)", fontsize=14, pad=15)
    plt.xlabel("Dimensione della Valanga $S$", fontsize=12)
    plt.ylabel("Probabilità Cumulativa $P(\\geq S)$", fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("figures/manna_powerlaw.png", dpi=300)
    print("Grafico salvato in figures/manna_powerlaw.png")
