from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="miplearn",
    version="0.2.0",
    author="Alinson S. Xavier",
    author_email="axavier@anl.gov",
    description="Extensible framework for Learning-Enhanced Mixed-Integer Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ANL-CEEESA/MIPLearn/",
    packages=find_namespace_packages(),
    python_requires=">=3.6",
    install_requires=[
        "docopt",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "p_tqdm",
        "pyomo",
        "python-markdown-math",
        "seaborn",
        "sklearn",
        "tqdm",
    ],
)
