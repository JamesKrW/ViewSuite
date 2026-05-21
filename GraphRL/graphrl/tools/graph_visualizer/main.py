#!/usr/bin/env python3
"""Graph Visualizer - Web frontend for exploring large graphs.

Usage:
    python main.py --graph_dir /path/to/graph --port 8050
"""

import argparse
import sys
import os

# Add parent to path so server package can be found
sys.path.insert(0, os.path.dirname(__file__))

from server.app import create_app


def main():
    parser = argparse.ArgumentParser(
        description="Graph Visualizer - Web frontend for exploring large graphs"
    )
    parser.add_argument(
        "--graph_dir", required=True,
        help="Path to graph directory containing graph.json and images/"
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port to serve on (default: 8050)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    # Validate graph_dir
    graph_json = os.path.join(args.graph_dir, "graph.json")
    if not os.path.exists(graph_json):
        print(f"Error: {graph_json} not found")
        sys.exit(1)

    app = create_app(args.graph_dir)
    print(f"\nServer ready at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
