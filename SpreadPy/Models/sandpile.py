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
    """

    def __init__(self, graph, sink=None, f=0, max_topplings=None):
        """
        :param graph:   networkx graph
        :param sink:    any node s ∈ V designated as sink (absorbs grains, never topples).
                        If None, no sink is used.
        :param f:       dissipation probability per grain transfer (default 0)
        :param max_topplings: safety guard M — max topplings per avalanche before
                              clipping excess grains. Default: 10 * number_of_nodes.
        """
        # the sandpile doesn't need retention, decay or suppress
        super(SandpileSpreading, self).__init__(graph, retention=0, decay=0, suppress=0)

        if sink is not None and sink not in self.graph.nodes:
            raise ValueError("Sink node must be in the graph.")

        self.sink = sink
        self.f = f
        # all nodes except the sink — these are the ones that can accumulate and topple
        self.non_sink_nodes = [n for n in self.graph.nodes if n != self.sink]
        # we cache the degree of each node so we don't recompute it every time
        self.node_degree = {n: self.graph.degree(n) for n in self.graph.nodes}
        # K_TOT: sum of degrees — also the max useful iteration count
        self.sum_degrees = sum(self.node_degree[n] for n in self.non_sink_nodes)
        # safety guard against divergent avalanches in the SOC chaotic regime
        self.max_topplings = max_topplings if max_topplings is not None else 10 * len(self.non_sink_nodes)

    def _add_grain(self, actual_status, node=None):
        """Add one grain to a single node. Returns the chosen node."""
        # if no node is specified, pick one at random among non-sink nodes
        if node is None:
            node = self.non_sink_nodes[np.random.randint(len(self.non_sink_nodes))]
        # give it +1 grain
        actual_status[node] += 1
        return node

    def _topple(self, actual_status):
        """Keep toppling unstable nodes until everything is stable again.

        If total topplings exceed self.max_topplings (SOC chaotic regime safeguard),
        toppling stops and diverged is set to True. The caller is responsible
        for restoring the pre-iteration state.
        """
        toppled_nodes = set()
        topplings_per_node = {}
        total_topplings = 0
        avalanche_size = 0
        grain_movements = []  # list of (from_node, to_node) for each grain moved
        diverged = False

        # a node is unstable when its grains > its degree (and degree > 0)
        unstable = [v for v in self.non_sink_nodes
                    if self.node_degree[v] > 0 and actual_status[v] > self.node_degree[v]]

        while unstable:
            # SOC safeguard: if topplings exceed M, signal divergence and stop
            if total_topplings >= self.max_topplings:
                diverged = True
                break

            for v in unstable:
                # normal node: it loses exactly deg(v) grains
                actual_status[v] -= self.node_degree[v]
                toppled_nodes.add(v)
                total_topplings += 1
                topplings_per_node[v] = topplings_per_node.get(v, 0) + 1
                avalanche_size += self.node_degree[v]

                # now distribute: each neighbour gets +1 grain
                for u in self.node_neighbors[v]:
                    if self.sink is not None and u == self.sink:
                        continue  # grain lost to sink
                    if self.f > 0 and np.random.random() < self.f:
                        continue  # grain dissipated stochastically
                    actual_status[u] += 1
                    grain_movements.append((v, u))

            # after this round of topples, check if new nodes became unstable
            unstable = [v for v in self.non_sink_nodes
                        if self.node_degree[v] > 0 and actual_status[v] > self.node_degree[v]]

        return {
            "toppled_nodes": toppled_nodes,  # insieme dei nodi che hanno superato la soglia almeno una volta
            "total_topplings": total_topplings,  # numero totale di eventi di toppling
            "unique_toppled_nodes": len(toppled_nodes), # quanti nodi distinti hanno superato la soglia almeno una volta
            "avalanche_size": avalanche_size, # grani che hanno *lasciato* i nodi toppled (= Σ deg(v)); nota: con sink o f>0 è > grani effettivamente arrivati (len(grain_movements))
            "topplings_per_node": topplings_per_node, # dizionario che associa a ogni nodo il numero di volte che ha superato la soglia
            "grain_movements": grain_movements, # lista di (from, to) per ogni grano spostato
            "diverged": diverged, # True se la cascata è stata interrotta dalla safeguard
        }

    def iteration(self, node=None, verbose=False):
        """
        Run one sandpile step: add grain(s) then topple until stable.

        :param node: where to drop the grain (random if None)
        :param verbose: if True, print debug info for this iteration
        :return: dict with 'iteration' number, 'status' snapshot, and avalanche stats
        """
        # take a snapshot of the current grain counts
        actual_status = {n: self.status[n] for n in self.graph.nodes}

        # iteration 0 is just the initial configuration, nothing happens yet
        if self.actual_iteration == 0:
            total_grains = sum(actual_status.values())
            self.actual_iteration += 1
            result = {"iteration": 0, "status": actual_status.copy(),
                    "toppled_nodes": set(), "total_topplings": 0,
                    "unique_toppled_nodes": 0, "avalanche_size": 0,
                    "topplings_per_node": {}, "grain_movements": [],
                    "has_avalanche": False, "diverged": False,
                    "grain_added_to": None, "total_grains": total_grains}
            if verbose:
                print(f"[iter 0] initial config — total_grains={total_grains}")
            return result

        # --- step 1: add grains ---
        # standard sandpile: just drop one grain on a single node
        grain_added_to = self._add_grain(actual_status, node=node)

        # --- step 2: topple until everything is stable ---
        avalanche = self._topple(actual_status)

        # --- step 2b: check divergence ---
        if avalanche["diverged"]:
            # Diverged: restore pre-iteration state, don't advance counter
            # The simulation should stop here
            total_grains = sum(self.status.values())
            iteration_num = self.actual_iteration
            if verbose:
                print(f"[iter {iteration_num}] grain -> {grain_added_to} | "
                      f"DIVERGED — simulation must stop (state NOT saved)")
            return {"iteration": iteration_num, "status": self.status.copy(),
                    **avalanche,
                    "has_avalanche": True,
                    "grain_added_to": grain_added_to,
                    "total_grains": total_grains}

        # --- step 2c: derived metrics ---
        has_avalanche = avalanche["total_topplings"] > 0
        total_grains = sum(actual_status.values())

        # --- step 3: save the new state ---
        self.status = actual_status
        self.actual_iteration += 1

        iteration_num = self.actual_iteration - 1

        if verbose:
            print(f"[iter {iteration_num}] grain -> {grain_added_to} | "
                  f"total_grains={total_grains} | "
                  f"avalanche={'YES' if has_avalanche else 'no'}")
            if has_avalanche:
                print(f"  toppled: {sorted(avalanche['toppled_nodes'])} | "
                      f"topplings={avalanche['total_topplings']} | "
                      f"movements={avalanche['grain_movements']}")

        return {"iteration": iteration_num, "status": actual_status.copy(),
                **avalanche,
                "has_avalanche": has_avalanche,
                "grain_added_to": grain_added_to,
                "total_grains": total_grains}