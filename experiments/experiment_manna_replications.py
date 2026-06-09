import pandas as pd
import networkx as nx
import numpy as np
import powerlaw
import os
import random

print("Caricamento rete SWOW per Manna Replications...")
df = pd.read_csv("data/SWOW-EN18/strength.SWOW-EN.R123.20180827.csv", sep='\t')
G_full = nx.from_pandas_edgelist(df, 'cue', 'response', edge_attr='R123.Strength', create_using=nx.DiGraph())

start_node = list(G_full.nodes())[0]
edges = list(nx.bfs_edges(G_full, start_node, depth_limit=3))
nodes = [start_node] + [v for u, v in edges]
nodes = nodes[:1000]
G = G_full.subgraph(nodes).copy()

routing_probs = {}
for n in G.nodes():
    out_edges = list(G.out_edges(n, data=True))
    if not out_edges:
        routing_probs[n] = None
    else:
        neighbors = [v for u, v, d in out_edges]
        weights = np.array([float(d.get('R123.Strength', 1.0)) for u, v, d in out_edges])
        if weights.sum() > 0:
            probs = weights / weights.sum()
        else:
            probs = np.ones(len(weights)) / len(weights)
        routing_probs[n] = (neighbors, probs)

sink_node = max(G.out_degree, key=lambda x: x[1])[0]
max_iters = 10000
alphas = []

print("Esecuzione di 5 replicazioni del Modello di Manna stocastico...")
for rep in range(5):
    status = {n: 0 for n in G.nodes()}
    avalanches = []
    
    for i in range(max_iters):
        target = random.choice(list(G.nodes()))
        status[target] += 1
        
        unstable = [n for n in G.nodes() if status[n] >= 2 and n != sink_node]
        avalanche_size = 0
        loop_safeguard = 0
        
        while unstable and loop_safeguard < 10000:
            for v in unstable:
                status[v] -= 2
                avalanche_size += 2
                
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
            
    if len(avalanches) > 10:
        results = powerlaw.Fit(avalanches, discrete=True, xmin=2, verbose=False)
        alphas.append(results.power_law.alpha)
        print(f"Rep {rep+1}: alpha = {results.power_law.alpha:.3f}")

mean_alpha = np.mean(alphas)
std_alpha = np.std(alphas)
print(f"\nManna Model Alpha: {mean_alpha:.3f} ± {std_alpha:.3f}")

with open("manna_stats.txt", "w") as f:
    f.write(f"Manna Model Alpha: {mean_alpha:.3f} +- {std_alpha:.3f}\n")
    f.write(f"Replications: {len(alphas)}\n")
