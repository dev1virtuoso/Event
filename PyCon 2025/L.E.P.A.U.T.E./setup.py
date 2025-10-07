# setup.py
from setuptools import setup, find_packages
import os

setup(
    name="lepaute",
    version="1.1.5",  # Incremented for feature extraction and evaluation fixes
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "kornia>=0.7.2",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torchvision>=0.15.0",
        "pytorch-metric-learning>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.8.0",
    ],
    author="Carson Wu",
    author_email="carson.developer1125@gmail.com",
    description="L.E.P.A.U.T.E. (Lie Equivariant Perception Algebraic Unified Transform Embedding Framework)",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/dev1virtuoso/Machine-Learning/tree/main/Computer%20Vision/L.E.P.A.U.T.E.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
)
