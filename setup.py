#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from setuptools import setup, find_namespace_packages

setup(
    name="miplearn",
    version="0.4.3",
    author="Alinson S. Xavier",
    author_email="axavier@anl.gov",
    description="Extensible Framework for Learning-Enhanced Mixed-Integer Optimization",
    url="https://github.com/ANL-CEEESA/MIPLearn/",
    packages=find_namespace_packages(),
    python_requires=">=3.9",
    install_requires=[
        "gurobipy>=12,<13",
        "h5py>=3,<4",
        "networkx>=2,<3",
        "numpy>=1,<2",
        "pandas>=2,<3",
        "pathos>=0.2,<0.3",
        "pyomo>=6,<7",
        "scikit-learn>=1,<2",
        "scipy>=1,<2",
        "tqdm>=4,<5",
    ],
    extras_require={
        "dev": [
            "Sphinx>=8,<9",
            "black==22.6.0",
            "mypy==1.8",
            "myst-parser>=4,<5",
            "nbsphinx>=0.9,<0.10",
            "pyflakes==2.5.0",
            "pytest>=7,<8",
            "sphinx-book-theme>=1,<2",
            "sphinx-multitoc-numbering==0.1.3",
            "twine>=6,<7",
            "ipython>=9,<10",
        ]
    },
)
