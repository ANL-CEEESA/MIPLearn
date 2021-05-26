#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="miplearn",
    version="0.2.0.dev9",
    author="Alinson S. Xavier",
    author_email="axavier@anl.gov",
    description="Extensible framework for Learning-Enhanced Mixed-Integer Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ANL-CEEESA/MIPLearn/",
    packages=find_namespace_packages(),
    python_requires=">=3.7",
    install_requires=[
        "matplotlib>=3,<4",
        "networkx>=2,<3",
        "numpy>=1,<1.21",
        "p_tqdm>=1,<2",
        "pandas>=1,<2",
        "pyomo>=5,<6",
        "pytest>=6,<7",
        "python-markdown-math>=0.8,<0.9",
        "seaborn>=0.11,<0.12",
        "scikit-learn>=0.24,<0.25",
        "tqdm>=4,<5",
        "mypy==0.790",
        "decorator>=4,<5",
        "overrides>=3,<4",
    ],
    extras_require={
        "dev": [
            "docopt>=0.6,<0.7",
            "black==20.8b1",
            "pre-commit>=2,<3",
            "pdoc3>=0.7,<0.8",
            "twine>=3,<4",
            "Sphinx>=3,<4",
            "sphinx-book-theme==0.1.0",
            "myst-parser==0.14.0",
        ]
    },
)
