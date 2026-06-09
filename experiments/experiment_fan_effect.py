"""
Effetto fan: figura fan_effect_sandpile.png + statistiche fan_effect_stats.txt
dallo STESSO run deterministico.

Protocollo corretto (probe non distruttivo): si raggiunge lo stato stazionario,
poi per ogni nodo bersaglio si misura la valanga innescata aggiungendo UN grano,
ripristinando lo stato del background dopo ogni prova. Cosi le misure per-nodo
sono campionate dallo stesso ensemble stazionario, senza la contaminazione
sequenziale della versione precedente.
"""
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from SpreadPy.Models.sandpile import SandpileSpreading
import os

np.random.seed(42)

print("Caricamento rete SWOW per Fan Effect...")
df = pd.read_csv("data/SWOW-EN18/strength.SWOW-EN.R123.20180827.csv", sep='\t')
G_full = nx.from_pandas_edgelist(df, 'cue', 'response')
largest_cc = max(nx.connected_components(G_full), key=len)
G_core = G_full.subgraph(largest_cc).copy()

start_node = list(G_core.nodes())[0]
edges = list(nx.bfs_edges(G_core, start_node, depth_limit=3))
nodes = ([start_node] + [v for u, v in edges])[:500]
G = G_core.subgraph(nodes).copy()
G.remove_edges_from(nx.selfloop_edges(G))

sink_node = max(G.degree, key=lambda x: x[1])[0]
degrees = dict(G.degree())
print("Calcolo betweenness...")
betweenness = nx.betweenness_centrality(G)

sp = SandpileSpreading(G, sink=sink_node, f=0, strict=True)
print("Burn-in...")
for _ in range(sp.sum_degrees):
    if sp.iteration()['diverged']:
        break

targets = [n for n in G.nodes() if n != sink_node and degrees[n] > 0]
K = 50
sums = {n: 0.0 for n in targets}
counts = {n: 0 for n in targets}

print("Probe non distruttivo...")
for trial in range(K):
    sp.iteration()                       # avanza il background con un grano casuale
    saved = dict(sp.status)              # snapshot dello stato stazionario corrente
    for node in targets:
        res = sp.iteration(node=node)    # prova: un grano sul bersaglio
        if not res['diverged']:
            sums[node] += res['avalanche_size']
            counts[node] += 1
        sp.status = dict(saved)          # ripristina il background (probe non distruttivo)

rows = [{'node': n, 'degree': degrees[n], 'betweenness': betweenness[n],
         'mean_avalanche': sums[n] / counts[n]} for n in targets if counts[n] > 0]
df_res = pd.DataFrame(rows)

corr_deg, p_deg = spearmanr(df_res['degree'], df_res['mean_avalanche'])
corr_bet, p_bet = spearmanr(df_res['betweenness'], df_res['mean_avalanche'])
corr_deg_bet, _ = spearmanr(df_res['degree'], df_res['betweenness'])
partial = (corr_bet - corr_deg * corr_deg_bet) / np.sqrt((1 - corr_deg**2) * (1 - corr_deg_bet**2))

# hub (quartile alto grado) vs periferia (quartile basso)
q_hi, q_lo = df_res['degree'].quantile(0.75), df_res['degree'].quantile(0.25)
hub = df_res[df_res['degree'] >= q_hi]['mean_avalanche']
peri = df_res[df_res['degree'] <= q_lo]['mean_avalanche']
ratio = hub.mean() / peri.mean() if peri.mean() > 0 else float('inf')
U, p_mw = mannwhitneyu(hub, peri, alternative='greater')

print(f"Spearman(Degree, Avalanche) = {corr_deg:.3f} (p={p_deg:.3e})")
print(f"Spearman(Betweenness, Avalanche) = {corr_bet:.3f} (p={p_bet:.3e})")
print(f"Partial (Betweenness, Avalanche | Degree) = {partial:.3f}")
print(f"ratio hub/periferia = {ratio:.1f}x   Mann-Whitney p={p_mw:.1e}")

# ---- figura ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.boxplot([hub.values, peri.values], labels=['High-fan\n(hub)', 'Low-fan\n(periferia)'])
ax1.set_ylabel(r'$\langle S \mid$ seed$\rangle$')
ax1.set_title(f"Effetto Fan\nMann-Whitney p={p_mw:.1e}  ratio={ratio:.1f}x", fontsize=11)
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

ax2.scatter(df_res['degree'], df_res['mean_avalanche'], color='#5b9bd5', alpha=0.6, edgecolors='white', linewidths=0.5)
z = np.polyfit(df_res['degree'], df_res['mean_avalanche'], 1)
xs = np.sort(df_res['degree'])
ax2.plot(xs, np.poly1d(z)(xs), color='#1f4e79',
         label=f'fit ($\\rho_s$={corr_deg:.2f}, p={p_deg:.1e})')
ax2.set_xlabel("grado del nodo  k")
ax2.set_ylabel(r'$\langle S \mid$ seed$\rangle$')
ax2.set_title("Grado x spreading (Spearman)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fan_effect_sandpile.png", dpi=200)

with open("fan_effect_stats.txt", "w") as fh:
    fh.write(f"Spearman(Degree, Avalanche) = {corr_deg:.3f} (p={p_deg:.3e})\n")
    fh.write(f"Spearman(Betweenness, Avalanche) = {corr_bet:.3f} (p={p_bet:.3e})\n")
    fh.write(f"Partial correlation (Betweenness, Avalanche | Degree) = {partial:.3f}\n")
    fh.write(f"ratio hub/periferia = {ratio:.1f}x (Mann-Whitney p={p_mw:.3e})\n")

print("\n===== NUMERI PER IL TESTO =====")
print(f"rho_s(degree)={corr_deg:.2f} p={p_deg:.1e}  partial={partial:.2f}  ratio={ratio:.0f}x")
