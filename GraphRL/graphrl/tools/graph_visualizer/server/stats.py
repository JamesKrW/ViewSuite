"""Statistics computation — pure graph-theoretic metrics only."""

import networkx as nx
from collections import Counter, defaultdict


class StatsEngine:
    def __init__(self, store):
        self.store = store
        self._cache = {}

    def global_stats(self) -> dict:
        """Global graph topology stats."""
        total_nodes = len(self.store.nodes)
        total_edges = len(self.store.edges)

        # Degree distribution (total = in + out)
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        for e in self.store.edges:
            out_deg[e["from"]] += 1
            in_deg[e["to"]] += 1

        total_deg = {nid: in_deg.get(nid, 0) + out_deg.get(nid, 0) for nid in self.store.nodes}
        avg_degree = sum(total_deg.values()) / max(len(total_deg), 1)
        deg_hist = Counter(total_deg.values())
        in_deg_hist = Counter(in_deg.values())
        out_deg_hist = Counter(out_deg.values())

        # Connected components (weakly connected, already computed)
        comp_sizes = [len(c) for c in self.store.components]

        # Strongly connected components (on the full directed graph)
        scc_stats = self._global_scc_stats()

        # Simple paths stats (just count isolated nodes, leaf nodes, etc.)
        isolated = sum(1 for d in total_deg.values() if d == 0)
        leaves = sum(1 for nid in self.store.nodes
                     if total_deg.get(nid, 0) == 1)
        sources = sum(1 for nid in self.store.nodes
                      if in_deg.get(nid, 0) == 0 and out_deg.get(nid, 0) > 0)
        sinks = sum(1 for nid in self.store.nodes
                    if out_deg.get(nid, 0) == 0 and in_deg.get(nid, 0) > 0)

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_components": len(self.store.components),
            "avg_degree": round(avg_degree, 2),
            "max_degree": max(total_deg.values()) if total_deg else 0,
            "max_in_degree": max(in_deg.values()) if in_deg else 0,
            "max_out_degree": max(out_deg.values()) if out_deg else 0,
            "isolated_nodes": isolated,
            "leaf_nodes": leaves,
            "source_nodes": sources,
            "sink_nodes": sinks,
            "degree_distribution": {str(k): v for k, v in sorted(deg_hist.items())[:30]},
            "in_degree_distribution": {str(k): v for k, v in sorted(in_deg_hist.items())[:30]},
            "out_degree_distribution": {str(k): v for k, v in sorted(out_deg_hist.items())[:30]},
            "component_sizes": comp_sizes[:100],
            **scc_stats,
        }

    def _global_scc_stats(self):
        """Strongly connected component stats for the whole graph."""
        cache_key = "global_scc"
        if cache_key in self._cache:
            return self._cache[cache_key]

        G = nx.DiGraph()
        G.add_nodes_from(self.store.nodes.keys())
        for e in self.store.edges:
            G.add_edge(e["from"], e["to"])

        sccs = list(nx.strongly_connected_components(G))
        sccs.sort(key=len, reverse=True)
        scc_sizes = [len(s) for s in sccs]

        # SCCs with more than 1 node (non-trivial)
        nontrivial_sccs = [s for s in scc_sizes if s > 1]

        result = {
            "total_strongly_connected_components": len(sccs),
            "nontrivial_scc_count": len(nontrivial_sccs),
            "largest_scc_size": scc_sizes[0] if scc_sizes else 0,
            "scc_sizes": scc_sizes[:100],
        }
        self._cache[cache_key] = result
        return result

    def component_stats(self, comp_idx: int) -> dict:
        """Graph-theoretic stats for a single connected component."""
        cache_key = f"comp_stats_{comp_idx}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if comp_idx < 0 or comp_idx >= len(self.store.components):
            return {"error": "Invalid component index"}

        nids = self.store.components[comp_idx]
        edge_indices = self.store.comp_edges.get(comp_idx, [])

        G = nx.DiGraph()
        G.add_nodes_from(nids)
        for eidx in edge_indices:
            e = self.store.edges[eidx]
            G.add_edge(e["from"], e["to"])

        # Degree stats
        degree_seq = [d for _, d in G.degree()]
        deg_hist = Counter(degree_seq)
        in_deg = [d for _, d in G.in_degree()]
        out_deg = [d for _, d in G.out_degree()]
        in_deg_hist = Counter(in_deg)
        out_deg_hist = Counter(out_deg)

        # Strongly connected components within this component
        sccs = list(nx.strongly_connected_components(G))
        sccs.sort(key=len, reverse=True)
        scc_sizes = [len(s) for s in sccs]
        nontrivial_sccs = [s for s in scc_sizes if s > 1]

        # Sources and sinks
        sources = sum(1 for _, d in G.in_degree() if d == 0)
        sinks = sum(1 for _, d in G.out_degree() if d == 0)

        # Diameter (only for small components, expensive for large ones)
        diameter = -1
        if len(nids) <= 5000:
            try:
                G_und = G.to_undirected()
                diameter = nx.diameter(G_und)
            except Exception:
                diameter = -1

        result = {
            "component_index": comp_idx,
            "node_count": len(nids),
            "edge_count": len(edge_indices),
            "avg_degree": round(sum(degree_seq) / max(len(degree_seq), 1), 2),
            "max_degree": max(degree_seq) if degree_seq else 0,
            "max_in_degree": max(in_deg) if in_deg else 0,
            "max_out_degree": max(out_deg) if out_deg else 0,
            "source_nodes": sources,
            "sink_nodes": sinks,
            "diameter": diameter,
            "strongly_connected_components": len(sccs),
            "nontrivial_scc_count": len(nontrivial_sccs),
            "largest_scc_size": scc_sizes[0] if scc_sizes else 0,
            "scc_sizes": scc_sizes[:50],
            "degree_distribution": {str(k): v for k, v in sorted(deg_hist.items())[:30]},
            "in_degree_distribution": {str(k): v for k, v in sorted(in_deg_hist.items())[:30]},
            "out_degree_distribution": {str(k): v for k, v in sorted(out_deg_hist.items())[:30]},
        }
        self._cache[cache_key] = result
        return result

    def components_table(self) -> list[dict]:
        """Summary table for all connected components."""
        cache_key = "comp_table"
        if cache_key in self._cache:
            return self._cache[cache_key]

        rows = []
        for cidx, comp in enumerate(self.store.components):
            n_nodes = len(comp)
            n_edges = len(self.store.comp_edges.get(cidx, []))
            rows.append({
                "index": cidx,
                "node_count": n_nodes,
                "edge_count": n_edges,
                "avg_degree": round(2 * n_edges / max(n_nodes, 1), 2),
            })
        self._cache[cache_key] = rows
        return rows

    def shortest_path(self, from_id: str, to_id: str):
        if from_id not in self.store.nodes or to_id not in self.store.nodes:
            return None

        comp_from = self.store.node_component.get(from_id, -1)
        comp_to = self.store.node_component.get(to_id, -1)

        if comp_from != comp_to:
            return {"path": [], "edges": [], "distance": -1,
                    "error": f"Nodes in different components ({comp_from} vs {comp_to})"}

        nids = self.store.components[comp_from]
        edge_indices = self.store.comp_edges.get(comp_from, [])

        G = nx.DiGraph()
        G.add_nodes_from(nids)
        edge_map = {}
        for eidx in edge_indices:
            e = self.store.edges[eidx]
            G.add_edge(e["from"], e["to"])
            edge_map[(e["from"], e["to"])] = e

        try:
            path = nx.shortest_path(G, from_id, to_id)
            edges = []
            for i in range(len(path) - 1):
                e = edge_map.get((path[i], path[i + 1]), {})
                edges.append({"from": path[i], "to": path[i + 1],
                              "action": e.get("obs_str", "")})
            return {"path": path, "edges": edges, "distance": len(path) - 1}
        except nx.NetworkXNoPath:
            return {"path": [], "edges": [], "distance": -1, "error": "No path found"}
        except nx.NodeNotFound as e:
            return {"path": [], "edges": [], "distance": -1, "error": str(e)}
