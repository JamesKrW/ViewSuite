"""REST API endpoints — supports multiple clustering methods."""

import os
from flask import Blueprint, request, jsonify, send_from_directory


def create_api_blueprint(store, stats, graph_dir):
    bp = Blueprint("api", __name__)
    images_dir = os.path.join(graph_dir, "images")

    @bp.route("/api/overview")
    def get_overview():
        limit = int(request.args.get("limit", 5000))
        return jsonify(store.get_overview(limit=limit))

    @bp.route("/api/clusters")
    def get_clusters():
        method = request.args.get("method", "wcc")
        field = request.args.get("field")
        clusters = store.get_clusters(method, field)
        return jsonify({"clusters": clusters, "total": len(clusters)})

    @bp.route("/api/cluster_graph")
    def get_cluster_graph():
        method = request.args.get("method", "wcc")
        idx = int(request.args.get("idx", 0))
        field = request.args.get("field")
        min_degree = int(request.args.get("min_degree", 0))
        result = store.get_cluster_graph(method, idx, field=field, min_degree=min_degree)
        return jsonify(result)

    @bp.route("/api/node/<node_id>")
    def get_node(node_id):
        detail = store.get_node_detail(node_id)
        if detail is None:
            return jsonify({"error": "Node not found"}), 404
        return jsonify(detail)

    @bp.route("/api/images/<path:filename>")
    def serve_image(filename):
        return send_from_directory(images_dir, filename, max_age=86400)

    @bp.route("/api/stats/global")
    def get_global_stats():
        return jsonify(stats.global_stats())

    @bp.route("/api/stats/component/<int:comp_idx>")
    def get_component_stats(comp_idx):
        return jsonify(stats.component_stats(comp_idx))

    @bp.route("/api/stats/components_table")
    def get_components_table():
        return jsonify({"components": stats.components_table()})

    @bp.route("/api/path/<from_id>/<to_id>")
    def get_path(from_id, to_id):
        result = stats.shortest_path(from_id, to_id)
        if result is None:
            return jsonify({"error": "Node not found"}), 404
        return jsonify(result)

    @bp.route("/api/search")
    def search():
        q = request.args.get("q", "")
        if len(q) < 2:
            return jsonify({"nodes": []})
        return jsonify({"nodes": store.search_nodes(q)})

    @bp.route("/api/layout_fields")
    def get_layout_fields():
        return jsonify({"fields": store.numeric_fields})

    @bp.route("/api/group_fields")
    def get_group_fields():
        return jsonify({"fields": store.group_fields})

    return bp
