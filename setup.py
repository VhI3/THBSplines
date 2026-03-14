"""
Minimal setuptools configuration for THBSplines.

All computation is done in pure Python / NumPy — no compiled extensions needed.
Install with:

    pip install -e .          # editable / development install
    conda env create -f environment.yml && conda activate thbsplines
"""

from setuptools import setup, find_packages

setup(
    name="THBSplines",
    version="0.2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "tqdm>=4.65",
    ],
    extras_require={
        "notebook": ["jupyterlab>=4.0", "ipympl>=0.9"],
        "dev":      ["pytest>=7.0", "pytest-cov"],
    },
    author="Ivar Stangeby",
    description=(
        "Truncated Hierarchical B-splines: adaptive refinement for "
        "isogeometric analysis, in pure Python."
    ),
    url="https://github.com/IvarStangeby/THBSplines",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
