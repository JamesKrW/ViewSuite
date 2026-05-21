"""
GraphRL environment packages.

Each environment is a Python package under graphrl/envs/ that registers
its components (env class, graph builder, SFT generator) on import.

To add a new environment, create a package here and import all components
in its __init__.py to trigger registration.
"""
