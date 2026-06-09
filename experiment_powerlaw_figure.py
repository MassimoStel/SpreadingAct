"""
Genera la figura a 2 pannelli soc_powerlaw_phase_transition.png E le statistiche
powerlaw_stats.txt DALLO STESSO run deterministico (seed fisso), cosi figura,
file statistiche e testo della tesi sono perfettamente coerenti.
Pannello (a): CCDF criticita' f=0 con fit a legge di potenza.
Pannello (b): dissipazione di bulk f>0 che tronca la coda.
"""
import pandas as pd
import networkx as nx
import numpy as np
import powerlaw
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
edges = list(nx.bfs_edges(G_core, start_node, depth_limit=4))
nodes = ([start_node] + [v for u, v in edges])[:1500]
G = G_core.subgraph(nodes).copy()
N_NODES = G.number_of_nodes()
print(f"Grafo: {N_NODES} nodi, {G.number_of_edges()} archi")

sink_node = max(G.degree, key=lambda x: x[1])[0]


def collect(f, n_iter):
    sp = SandpileSpreading(G, sink=sink_node, f=f, strict=True)
    for _ in range(sp.sum_degrees):           # burn-in
        if sp.iteration()['diverged']:
            break
    avs = []
    for _ in range(n_iter):
        res = sp.iteration()
        if res['diverged']:
            break
        if res['has_avalanche']:
            avs.append(res['avalanche_size'])
    return np.array(avs)


def ccdf(data):
    x = np.unique(data)
    y = np.array([(data >= xi).mean() for xi in x])
    return x, y


# ---- Pannello (a): criticita' f=0 ----
print("Run f=0 ...")
avs0 = collect(0.0, 25000)
print(f"Valanghe f=0: {len(avs0)}")

fit = powerlaw.Fit(avs0, discrete=True)
alpha = fit.power_law.alpha
xmin = fit.xmin
sigma = fit.power_law.sigma
ci_lo, ci_hi = alpha - 1.96 * sigma, alpha + 1.96 * sigma
R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
R_tp, p_tp = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)
R_ex, p_ex = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)

# ---- Pannello (b): sweep di dissipazione ----
f_values = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
sweep = {}
for f in f_values:
    if f == 0.0:
        sweep[f] = avs0
    else:
        print(f"Run f={f} ...")
        sweep[f] = collect(f, 10000)

# ---- Figura ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

fit.plot_pdf(ax=ax1, color='#1f77b4', linewidth=1.6, label=f'f=0 (N={len(avs0)} valanghe)')
fit.power_law.plot_pdf(ax=ax1, color='k', linestyle='--', linewidth=1.5,
                       label=f'fit legge di potenza\n$\\alpha$={alpha:.2f}, $x_{{min}}$={xmin:.0f}')
ax1.set_title(f"Criticita' (f=0): legge di potenza su {N_NODES} nodi\n"
              f"$\\alpha$={alpha:.2f}  ($x_{{min}}$={xmin:.0f})  "
              f"LR(pl vs exp)={R_ex:.1f}, p={p_ex:.1e}", fontsize=11)
ax1.set_xlabel("Dimensione della valanga  $S$")
ax1.set_ylabel(r"$P(S)$  (PDF log-binned)")
ax1.legend(fontsize=9)
ax1.grid(True, which="both", linestyle='--', alpha=0.3)

cmap = plt.cm.viridis(np.linspace(0, 0.9, len(f_values)))
for f, c in zip(f_values, cmap):
    avs = sweep[f]
    if len(avs) == 0:
        continue
    powerlaw.plot_pdf(avs, ax=ax2, color=c, linewidth=1.4, linear_bins=False,
                      label=f"f={f:.2f}  $\\langle S \\rangle$={avs.mean():.0f}")
ax2.set_title("Dissipazione di bulk: $f$ tronca la coda\n(transizione critico $\\rightarrow$ subcritico)", fontsize=11)
ax2.set_xlabel("Dimensione della valanga  $S$")
ax2.set_ylabel(r"$P(S)$  (PDF log-binned)")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", linestyle='--', alpha=0.3)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/soc_powerlaw_phase_transition.png", dpi=200)
print("Figura salvata.")

# ---- stats ----
with open("powerlaw_stats.txt", "w") as fh:
    fh.write(f"N nodi: {N_NODES}\n")
    fh.write(f"xmin stimato: {xmin}\n")
    fh.write(f"alpha: {alpha:.4f}\n")
    fh.write(f"Sigma: {sigma:.4f}\n")
    fh.write(f"CI 95% per alpha: [{ci_lo:.4f}, {ci_hi:.4f}]\n")
    fh.write(f"Power-law vs Lognormal: R={R_ln:.4f}, p={p_ln:.4f}\n")
    fh.write(f"Power-law vs Truncated Power-law: R={R_tp:.4f}, p={p_tp:.4f}\n")
    fh.write(f"Power-law vs Exponential: R={R_ex:.4f}, p={p_ex:.4f}\n")
    fh.write(f"N valanghe (f=0): {len(avs0)}\n")

print("\n===== NUMERI PER IL TESTO =====")
print(f"N_NODI={N_NODES}  N_VAL={len(avs0)}")
print(f"alpha={alpha:.2f}  xmin={xmin:.0f}  CI=[{ci_lo:.2f},{ci_hi:.2f}]  alpha-1={alpha-1:.2f}")
print(f"lognormal R={R_ln:.2f} p={p_ln:.2f}")
print(f"truncated R={R_tp:.2f} p={p_tp:.3f}")
print(f"exponential R={R_ex:.2f} p={p_ex:.1e}")
print("S medie sweep:", {f: round(float(sweep[f].mean()), 0) if len(sweep[f]) else None for f in f_values})
