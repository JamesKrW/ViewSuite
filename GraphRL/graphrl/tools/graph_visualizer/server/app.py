"""Flask application factory."""

import os
from flask import Flask
from .graph_store import GraphStore
from .stats import StatsEngine


def create_app(graph_dir: str) -> Flask:
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    app = Flask(__name__, static_folder=static_dir, static_url_path="/static")

    # Load graph data
    print(f"Loading graph from {graph_dir}...")
    store = GraphStore(graph_dir)
    stats = StatsEngine(store)

    # Store in app config for access by routes
    app.config["graph_store"] = store
    app.config["stats_engine"] = stats
    app.config["graph_dir"] = graph_dir

    # Register routes
    from .routes_api import create_api_blueprint
    app.register_blueprint(create_api_blueprint(store, stats, graph_dir))

    # Serve index.html at root
    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    return app
