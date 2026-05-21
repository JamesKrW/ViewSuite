from setuptools import setup, find_packages

setup(
    name="graphrl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "networkx>=3.0",
        "trl==0.26.2",
        "deepspeed>=0.10.0,<=0.18.4",
    ],
)
