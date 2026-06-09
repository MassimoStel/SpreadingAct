from ..SpreadingModel import SpreadingBaseModel
import numpy as np

__all__ = ["SandpileSpreading"]


class SandpileSpreading(SpreadingBaseModel):
    """
    Abelian Sandpile Model on arbitrary graphs.

    Following the generalization by Dhar (1990), the sandpile is defined on
    any undirected finite graph G = (V, E). A designated vertex s ∈ V, called
    the sink, is not allowed to topple: grains sent to it are lost (dissipation).
    Any vertex can serve as the sink — no virtual nodes are needed.

    Each iteration drops 1 grain on a random non-sink node and topples
    until stable. This is the classic BTW sandpile on arbitrary graphs.

    Optionally, a dissipation probability f can be set: every grain that
    moves from one node to another disappears with probability f instead
    of arriving. When f > 0 and no sink is specified, dissipation is
    entirely stochastic.

    Notes (for rigour):
    - The model is *deterministic and abelian* only for f = 0 (with or without
      a sink). For f > 0 the dynamics become stochastic (a Manna-type
      dissipative sandpile), so the final configuration is no longer
      order-independent. The power-law / criticality analysis therefore uses
      f = 0; f > 0 is used only to show the controlled tail cutoff.
    - Avalanche size S = sum of deg(v) over all topplings = grains *sent*. It is
      exact at f = 0; for f > 0 it counts attempted transfers (some dissipated).
    - Only the graph *topology* matters: a node always sends one grain per
      neighbour. Edge weights, if present, are ignored.
    """

    def __init__(self, graph, sink=None, f=0, strict=True, max_topplings=None):
        """
        :param graph:   networkx graph
        :param sink:    any node s ∈ V designated as sink (absorbs grains, never topples).
                        If None, no sink is used.
        :param f:       dissipation probability per grain transfer (default 0)
        :param strict:  toppling-threshold convention.
                        True  -> a node topples when height >  degree
                                 (max stable height = degree). Default.
                        False -> a node topples when height >= degree
                                 (classic BTW/Dhar, max stable height = degree - 1).
        :param max_topplings: safety guard M — max topplings per avalanche before
                              the cascade is declared divergent. Default: 10 * number_of_nodes.
        """
        # the sandpile doesn't need retention, decay or suppress
        super(SandpileSpreading, self).__init__(graph, retention=0, decay=0, suppress=0)

        if sink is not None and sink not in self.graph.nodes:
            raise ValueError("Sink node must be in the graph.")

        self.sink = sink
        self.f = f
        self.strict = strict
        # all nodes except the sink — these are the ones that can accumulate and topple
        self.non_sink_nodes = [n for n in self.graph.nodes if n != self.sink]
        # we cache the degree of each node so we don't recompute it every time
        self.node_degree = {n: self.graph.degree(n) for n in self.graph.nodes}
        # critical height = lowest height at which a node becomes unstable.
        #   strict '>' : unstable when height >  degree  -> crit = degree + 1
        #   classic '>=': unstable when height >= degree -> crit = degree
        self.crit = {n: self.node_degree[n] + (1 if strict else 0)
                     for n in self.graph.nodes}
        # K_TOT: sum of degrees over non-sink nodes (relaxation/burn-in time, Dhar 1990)
        self.sum_degrees = sum(self.node_degree[n] for n in self.non_sink_nodes)
        # Maximum stable mass: total grains the system can hold in a stable config,
        # = sum of (crit - 1) over non-sink nodes with degree > 0.
        # In a CLOSED system (sink=None, f=0) no stable configuration can hold more
        # than this; once total_grains exceeds it the dynamics can no longer terminate
        # -> supercritical/"chaotic" regime. Use it as the order-parameter ceiling.
        self.max_stable_mass = sum(self.crit[n] - 1
                                   for n in self.non_sink_nodes
                                   if self.node_degree[n] > 0)
        # safety guard against divergent avalanches in the supercritical regime
        self.max_topplings = max_topplings if max_topplings is not None else 10 * len(self.non_sink_nodes)

    def _add_grain(self, actual_status, node=None):
        """Add one grain to a single non-sink node. Returns the chosen node."""
        if node is None:
            node = self.non_sink_nodes[np.random.randint(len(self.non_sink_nodes))]
        elif self.sink is not None and node == self.sink:
            raise ValueError("Cannot add a grain to the sink node "
                             "(grains added to the sink would be lost silently).")
        actual_status[node] += 1
        return node

    def _unstable(self, status):
        """Nodes that reach the toppling threshold (height >= crit; see `strict`)."""
        return [v for v in self.non_sink_nodes
                if self.node_degree[v] > 0 and status[v] >= self.crit[v]]

    @staticmethod
    def _empty_avalanche():
        """Empty avalanche dict — used for iteration 0 and for diverged steps."""
        return {"toppled_nodes": set(), "total_topplings": 0,
                "unique_toppled_nodes": 0, "avalanche_size": 0,
                "topplings_per_node": {}, "grain_movements": [],
                "diverged": False}

    def _topple(self, actual_status):
        """Topple all unstable nodes (parallel rounds) until the system is stable.

        If total topplings exceed `self.max_topplings` (SOC chaotic regime safeguard),
        toppling stops and `diverged=True` is returned. The caller is responsible
        for restoring the pre-iteration state.
        """
        toppled_nodes = set()
        topplings_per_node = {}
        total_topplings = 0
        avalanche_size = 0
        grain_movements = []
        diverged = False

        unstable = self._unstable(actual_status)

        while unstable:
            if total_topplings >= self.max_topplings:
                diverged = True
                break

            for v in unstable:
                deg_v = self.node_degree[v]
                actual_status[v] -= deg_v
                toppled_nodes.add(v)
                total_topplings += 1
                topplings_per_node[v] = topplings_per_node.get(v, 0) + 1
                avalanche_size += deg_v

                for u in self.node_neighbors[v]:
                    if self.sink is not None and u == self.sink:
                        continue  # grain absorbed by sink
                    if self.f > 0 and np.random.random() < self.f:
                        continue  # grain dissipated stochastically
                    actual_status[u] += 1
                    grain_movements.append((v, u))

            unstable = self._unstable(actual_status)

        return {"toppled_nodes": toppled_nodes,
                "total_topplings": total_topplings,
                "unique_toppled_nodes": len(toppled_nodes),
                "avalanche_size": avalanche_size,
                "topplings_per_node": topplings_per_node,
                "grain_movements": grain_movements,
                "diverged": diverged}

    def iteration(self, node=None, verbose=False):
        """Run one sandpile step: add a grain, topple until stable.

        :param node: where to drop the grain (random non-sink node if None)
        :param verbose: print debug info if True
        :return: dict with iteration number, status snapshot and avalanche stats
        """
        # iteration 0 = initial configuration, no grain dropped
        if self.actual_iteration == 0:
            self.actual_iteration += 1
            if verbose:
                print(f"[iter 0] initial config — total_grains={sum(self.status.values())}")
            return {"iteration": 0, "status": self.status.copy(),
                    "has_avalanche": False, "grain_added_to": None,
                    "total_grains": sum(self.status.values()),
                    **self._empty_avalanche()}

        actual_status = {n: self.status[n] for n in self.graph.nodes}
        grain_added_to = self._add_grain(actual_status, node=node)
        avalanche = self._topple(actual_status)

        # Diverged: cascade aborted, state NOT advanced, counter NOT incremented.
        # We return CLEAN empty stats — the caller must check `diverged` and stop.
        if avalanche["diverged"]:
            iteration_num = self.actual_iteration
            if verbose:
                print(f"[iter {iteration_num}] grain -> {grain_added_to} | "
                      f"DIVERGED — simulation must stop (state NOT saved)")
            return {"iteration": iteration_num, "status": self.status.copy(),
                    "has_avalanche": False, "grain_added_to": grain_added_to,
                    "total_grains": sum(self.status.values()),
                    **self._empty_avalanche(), "diverged": True}

        # Success: commit new state
        self.status = actual_status
        self.actual_iteration += 1
        iteration_num = self.actual_iteration - 1
        has_avalanche = avalanche["total_topplings"] > 0

        if verbose:
            total_grains = sum(actual_status.values())
            print(f"[iter {iteration_num}] grain -> {grain_added_to} | "
                  f"total_grains={total_grains} | "
                  f"avalanche={'YES' if has_avalanche else 'no'}")
            if has_avalanche:
                print(f"  toppled: {sorted(avalanche['toppled_nodes'])} | "
                      f"topplings={avalanche['total_topplings']} | "
                      f"movements={avalanche['grain_movements']}")

        return {"iteration": iteration_num, "status": actual_status.copy(),
                "has_avalanche": has_avalanche, "grain_added_to": grain_added_to,
                "total_grains": sum(actual_status.values()),
                **avalanche}