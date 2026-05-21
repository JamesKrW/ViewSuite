"""Graph data loading, indexing, and in-memory store.

Generic graph store — no domain-specific assumptions.
Supports multiple clustering methods: WCC, SCC, Group By Field.
"""

import json
import math
import os
import pickle
import re
import time
from collections import defaultdict

import networkx as nx


class GraphStore:
    def __init__(self, graph_dir: str):
        self.graph_dir = graph_dir
        self.graph_json_path = os.path.join(graph_dir, "graph.json")
        self.images_dir = os.path.join(graph_dir, "images")
        self.cache_path = os.path.join(graph_dir, ".graph_cache_v3.pkl")

        # Raw data
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []

        # Adjacency
        self.node_out_edges: dict[str, list[int]] = {}
        self.node_in_edges: dict[str, list[int]] = {}

        # Connected components (WCC, sorted by size descending)
        self.components: list[list[str]] = []
        self.node_component: dict[str, int] = {}
        self.comp_edges: dict[int, list[int]] = {}

        # Strongly connected components (sorted by size descending)
        self.sccs: list[list[str]] = []
        self.node_scc: dict[str, int] = {}
        self.scc_edges: dict[int, list[int]] = {}

        # Auto-detected fields
        self.numeric_fields: list[dict] = []
        self.group_fields: list[dict] = []

        # Derived (not cached)
        self._group_cache: dict = {}

        self._load()

    # ── Loading ──────────────────────────────────────────────────────

    def _load(self):
        if self._try_load_cache():
            return
        self._load_json()
        self._build_indices()
        self._save_cache()

    def _try_load_cache(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        json_mtime = os.path.getmtime(self.graph_json_path)
        cache_mtime = os.path.getmtime(self.cache_path)
        if cache_mtime < json_mtime:
            print("[cache] Cache older than graph.json, rebuilding...")
            return False
        try:
            t0 = time.time()
            print("[cache] Loading from pickle cache...")
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            for key in [
                "nodes", "edges", "node_out_edges", "node_in_edges",
                "components", "node_component", "comp_edges",
                "sccs", "node_scc", "scc_edges",
                "numeric_fields", "group_fields",
            ]:
                setattr(self, key, data[key])
            print(f"[cache] Loaded in {time.time() - t0:.1f}s "
                  f"({len(self.nodes)} nodes, {len(self.edges)} edges, "
                  f"{len(self.components)} WCC, {len(self.sccs)} SCC)")
            return True
        except Exception as e:
            print(f"[cache] Failed to load cache: {e}")
            return False

    def _load_json(self):
        t0 = time.time()
        print(f"[load] Loading {self.graph_json_path} ...")
        with open(self.graph_json_path, "r") as f:
            raw = f.read()
        print(f"[load] Read file in {time.time() - t0:.1f}s ({len(raw) / 1e6:.0f}MB)")

        t1 = time.time()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print("[load] Standard JSON failed, fixing trailing commas...")
            fixed = re.sub(r',\s*([}\]])', r'\1', raw)
            data = json.loads(fixed)
        del raw
        print(f"[load] Parsed JSON in {time.time() - t1:.1f}s")

        self.nodes = {}
        for nid, ndata in data.get("nodes", {}).items():
            self.nodes[nid] = {
                "state": ndata.get("state", {}),
                "obs_str": ndata.get("obs_str", ndata.get("text", "")),
                "image_paths": ndata.get("image_paths", ndata.get("images", [])),
                "extra": ndata.get("extra", {}),
            }

        self.edges = []
        for edata in data.get("edges", []):
            self.edges.append({
                "from": edata.get("from", ""),
                "to": edata.get("to", ""),
                "obs_str": edata.get("obs_str", edata.get("text", "")),
                "image_paths": edata.get("image_paths", edata.get("images", [])),
                "extra": edata.get("extra", {}),
            })
        del data
        print(f"[load] Loaded {len(self.nodes)} nodes, {len(self.edges)} edges")

    def _build_indices(self):
        t0 = time.time()
        print("[index] Building indices...")

        # Adjacency
        node_out = defaultdict(list)
        node_in = defaultdict(list)
        for idx, edge in enumerate(self.edges):
            node_out[edge["from"]].append(idx)
            node_in[edge["to"]].append(idx)
        self.node_out_edges = dict(node_out)
        self.node_in_edges = dict(node_in)

        # WCC via NetworkX
        print("[index] Computing connected components (WCC)...")
        G = nx.Graph()
        G.add_nodes_from(self.nodes.keys())
        for e in self.edges:
            if e["from"] in self.nodes and e["to"] in self.nodes:
                G.add_edge(e["from"], e["to"])
        raw_components = list(nx.connected_components(G))
        raw_components.sort(key=len, reverse=True)
        del G

        self.components = [sorted(list(c)) for c in raw_components]
        self.node_component = {}
        for cidx, comp in enumerate(self.components):
            for nid in comp:
                self.node_component[nid] = cidx

        comp_edges_map = defaultdict(list)
        for idx, e in enumerate(self.edges):
            cidx = self.node_component.get(e["from"])
            if cidx is not None:
                comp_edges_map[cidx].append(idx)
        self.comp_edges = dict(comp_edges_map)

        # SCC via NetworkX
        print("[index] Computing strongly connected components (SCC)...")
        DG = nx.DiGraph()
        DG.add_nodes_from(self.nodes.keys())
        for e in self.edges:
            if e["from"] in self.nodes and e["to"] in self.nodes:
                DG.add_edge(e["from"], e["to"])
        raw_sccs = list(nx.strongly_connected_components(DG))
        raw_sccs.sort(key=len, reverse=True)
        del DG

        self.sccs = [sorted(list(c)) for c in raw_sccs]
        self.node_scc = {}
        for sidx, scc in enumerate(self.sccs):
            for nid in scc:
                self.node_scc[nid] = sidx

        scc_edges_map = defaultdict(list)
        for idx, e in enumerate(self.edges):
            src_scc = self.node_scc.get(e["from"])
            if src_scc is not None and self.node_scc.get(e["to"]) == src_scc:
                scc_edges_map[src_scc].append(idx)
        self.scc_edges = dict(scc_edges_map)

        # Auto-detect fields
        self._detect_numeric_fields()
        self._detect_group_fields()

        non_trivial_sccs = sum(1 for c in self.sccs if len(c) >= 2)
        print(f"[index] Built indices in {time.time() - t0:.1f}s "
              f"({len(self.components)} WCC, {non_trivial_sccs} non-trivial SCC)")

    def _detect_numeric_fields(self):
        sample_size = min(100, len(self.nodes))
        sample_nids = list(self.nodes.keys())[:sample_size]
        field_values = defaultdict(list)

        def _walk(obj, prefix):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _walk(v, f"{prefix}.{k}" if prefix else k)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                field_values[prefix].append(obj)

        for nid in sample_nids:
            ndata = self.nodes[nid]
            _walk(ndata.get("state", {}), "state")
            _walk(ndata.get("extra", {}), "extra")

        self.numeric_fields = []
        for path, vals in field_values.items():
            if len(vals) < sample_size * 0.5:
                continue
            vmin, vmax = min(vals), max(vals)
            if vmax - vmin < 1e-9:
                continue
            self.numeric_fields.append({
                "path": path, "label": path.split(".")[-1],
                "min": vmin, "max": vmax,
            })
        print(f"[index] Detected {len(self.numeric_fields)} numeric fields")

    def _detect_group_fields(self):
        sample_size = min(200, len(self.nodes))
        sample_nids = list(self.nodes.keys())[:sample_size]
        field_values = defaultdict(set)

        def _walk(obj, prefix):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _walk(v, f"{prefix}.{k}" if prefix else k)
            elif isinstance(obj, str) and obj:
                field_values[prefix].add(obj)

        for nid in sample_nids:
            ndata = self.nodes[nid]
            _walk(ndata.get("state", {}), "state")
            _walk(ndata.get("extra", {}), "extra")

        self.group_fields = []
        for path, vals in field_values.items():
            if len(vals) < 2:
                continue
            if len(vals) > sample_size * 0.9:
                continue
            self.group_fields.append({"path": path, "label": path.split(".")[-1]})

        for f in self.numeric_fields:
            self.group_fields.append({"path": f["path"], "label": f["label"]})
        print(f"[index] Detected {len(self.group_fields)} group-by fields")

    def _save_cache(self):
        try:
            t0 = time.time()
            data = {key: getattr(self, key) for key in [
                "nodes", "edges", "node_out_edges", "node_in_edges",
                "components", "node_component", "comp_edges",
                "sccs", "node_scc", "scc_edges",
                "numeric_fields", "group_fields",
            ]}
            with open(self.cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[cache] Saved cache in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"[cache] Failed to save cache: {e}")

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_nested(obj, path: str):
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                return None
        return obj

    def _get_field_value(self, nid: str, field_path: str):
        ndata = self.nodes.get(nid)
        if not ndata:
            return None
        parts = field_path.split(".", 1)
        if len(parts) == 2:
            return self._get_nested(ndata.get(parts[0], {}), parts[1])
        return ndata.get(parts[0])

    def _compute_groups(self, field: str):
        """Group ALL nodes by field value. Returns [(value, [nids])] sorted by size desc."""
        if field in self._group_cache:
            return self._group_cache[field]

        groups = defaultdict(list)
        for nid in self.nodes:
            val = self._get_field_value(nid, field)
            val = str(val) if val is not None else ""
            groups[val].append(nid)

        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
        self._group_cache[field] = sorted_groups
        return sorted_groups

    # ── Cluster listing ──────────────────────────────────────────────

    def get_clusters(self, method='wcc', field=None):
        """Return list of cluster summaries for the dropdown."""
        if method == 'wcc':
            return [{
                "index": i, "label": f"#{i}",
                "node_count": len(c),
                "edge_count": len(self.comp_edges.get(i, [])),
            } for i, c in enumerate(self.components)]

        elif method == 'scc':
            result = []
            for i, c in enumerate(self.sccs):
                if len(c) < 2:
                    break  # sorted desc, rest are trivial
                result.append({
                    "index": len(result), "orig_index": i,
                    "label": f"SCC #{len(result)}",
                    "node_count": len(c),
                    "edge_count": len(self.scc_edges.get(i, [])),
                })
            return result

        elif method == 'group' and field:
            groups = self._compute_groups(field)
            return [{
                "index": i,
                "label": val if val else "(empty)",
                "node_count": len(nids),
            } for i, (val, nids) in enumerate(groups)]

        return []

    # ── Overview (LOD) ──────────────────────────────────────────────

    def get_overview(self, limit=5000):
        """Return top N nodes by degree for the 'All Graph' LOD view."""
        # Sort nodes by total degree descending
        node_degrees = []
        for nid in self.nodes:
            deg = len(self.node_out_edges.get(nid, [])) + len(self.node_in_edges.get(nid, []))
            node_degrees.append((nid, deg))
        node_degrees.sort(key=lambda x: -x[1])

        top_nids = set(nid for nid, _ in node_degrees[:limit])

        # Collect edges within top nodes using adjacency
        edge_indices = []
        for nid in top_nids:
            for eidx in self.node_out_edges.get(nid, []):
                if self.edges[eidx]["to"] in top_nids:
                    edge_indices.append(eidx)

        return self._build_subgraph(top_nids, edge_indices)

    # ── Cluster graph ────────────────────────────────────────────────

    def get_cluster_graph(self, method, cluster_idx, field=None, min_degree=0):
        """Get the actual subgraph for a specific cluster."""
        if method == 'wcc':
            return self._build_subgraph(
                set(self.components[cluster_idx]) if cluster_idx < len(self.components) else set(),
                self.comp_edges.get(cluster_idx, []),
                min_degree,
            )

        elif method == 'scc':
            # Map from display index to original SCC index
            non_trivial = [i for i, c in enumerate(self.sccs) if len(c) >= 2]
            if cluster_idx >= len(non_trivial):
                return {"nodes": [], "edges": []}
            orig_idx = non_trivial[cluster_idx]
            return self._build_subgraph(
                set(self.sccs[orig_idx]),
                self.scc_edges.get(orig_idx, []),
                min_degree,
            )

        elif method == 'group' and field:
            groups = self._compute_groups(field)
            if cluster_idx >= len(groups):
                return {"nodes": [], "edges": []}
            _, nids_list = groups[cluster_idx]
            nids = set(nids_list)
            # Collect internal edges via adjacency
            edge_indices = []
            for nid in nids:
                for eidx in self.node_out_edges.get(nid, []):
                    if self.edges[eidx]["to"] in nids:
                        edge_indices.append(eidx)
            return self._build_subgraph(nids, edge_indices, min_degree)

        return {"nodes": [], "edges": []}

    def _build_subgraph(self, nids: set, edge_indices: list, min_degree: int = 0):
        """Build a standard subgraph response with WCC/SCC stats."""
        if not nids:
            return {"nodes": [], "edges": [], "num_sub_components": 0,
                    "sub_component_sizes": [], "num_sccs": 0, "scc_sizes": []}

        # Compute degree
        degree = defaultdict(int)
        for eidx in edge_indices:
            e = self.edges[eidx]
            if e["from"] in nids and e["to"] in nids:
                degree[e["from"]] += 1
                degree[e["to"]] += 1

        # Min degree filter
        if min_degree > 0:
            nids = {nid for nid in nids if degree.get(nid, 0) >= min_degree}
            edge_indices = [eidx for eidx in edge_indices
                            if self.edges[eidx]["from"] in nids and self.edges[eidx]["to"] in nids]
            degree = defaultdict(int)
            for eidx in edge_indices:
                e = self.edges[eidx]
                degree[e["from"]] += 1
                degree[e["to"]] += 1

        # WCC within subgraph
        G = nx.Graph()
        G.add_nodes_from(nids)
        for eidx in edge_indices:
            e = self.edges[eidx]
            G.add_edge(e["from"], e["to"])
        sub_comps = list(nx.connected_components(G))
        sub_comps.sort(key=len, reverse=True)
        node_sub_comp = {}
        for sci, sc in enumerate(sub_comps):
            for nid in sc:
                node_sub_comp[nid] = sci
        del G

        # SCC within subgraph
        DG = nx.DiGraph()
        DG.add_nodes_from(nids)
        for eidx in edge_indices:
            e = self.edges[eidx]
            DG.add_edge(e["from"], e["to"])
        sccs = list(nx.strongly_connected_components(DG))
        sccs.sort(key=len, reverse=True)
        node_scc = {}
        for sci, sc in enumerate(sccs):
            for nid in sc:
                node_scc[nid] = sci
        del DG

        # Build response
        nodes_out = []
        for nid in nids:
            ndata = self.nodes[nid]
            node_out = {
                "id": nid,
                "degree": degree.get(nid, 0),
                "sub_component": node_sub_comp.get(nid, 0),
                "scc_id": node_scc.get(nid, 0),
            }
            for field in self.numeric_fields:
                val = self._get_field_value(nid, field["path"])
                node_out[field["path"]] = val if val is not None else 0
            for field in self.group_fields:
                if field["path"] not in node_out:
                    val = self._get_field_value(nid, field["path"])
                    node_out[field["path"]] = val if val is not None else ""
            nodes_out.append(node_out)

        edges_out = []
        for eidx in edge_indices:
            e = self.edges[eidx]
            if e["from"] in nids and e["to"] in nids:
                edges_out.append({
                    "id": eidx, "source": e["from"],
                    "target": e["to"], "action": e["obs_str"],
                })

        return {
            "nodes": nodes_out, "edges": edges_out,
            "num_sub_components": len(sub_comps),
            "sub_component_sizes": [len(c) for c in sub_comps[:50]],
            "num_sccs": len(sccs),
            "scc_sizes": [len(c) for c in sccs[:50]],
        }

    # ── Node detail & search ─────────────────────────────────────────

    def get_node_detail(self, node_id: str):
        if node_id not in self.nodes:
            return None
        ndata = self.nodes[node_id]
        out_edges = [{"id": eidx, "to": self.edges[eidx]["to"],
                       "action": self.edges[eidx]["obs_str"]}
                     for eidx in self.node_out_edges.get(node_id, [])]
        in_edges = [{"id": eidx, "from": self.edges[eidx]["from"],
                      "action": self.edges[eidx]["obs_str"]}
                    for eidx in self.node_in_edges.get(node_id, [])]
        return {
            "id": node_id,
            "state": ndata.get("state", {}),
            "obs_str": ndata.get("obs_str", ""),
            "image_paths": ndata.get("image_paths", []),
            "extra": ndata.get("extra", {}),
            "component": self.node_component.get(node_id, -1),
            "out_edges": out_edges, "in_edges": in_edges,
            "out_degree": len(out_edges), "in_degree": len(in_edges),
        }

    def search_nodes(self, query: str, limit: int = 50):
        query_lower = query.lower()
        results = []
        for nid, ndata in self.nodes.items():
            if query_lower in nid.lower() or query_lower in ndata.get("obs_str", "").lower():
                results.append({
                    "id": nid,
                    "component": self.node_component.get(nid, -1),
                    "obs_str": ndata.get("obs_str", "")[:100],
                })
                if len(results) >= limit:
                    break
        return results
