from setuptools import setup, find_namespace_packages

setup(
    name='miplearn',
    version='0.1',
    description='A Machine-Learning Framework for Mixed-Integer Optimization',
    author='Alinson S. Xavier',
    author_email='axavier@anl.gov',
    packages=find_namespace_packages(),
    install_requires=[
        'docopt',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'p_tqdm',
        'pyomo',
        'python-markdown-math',
        'seaborn',
        'sklearn',
        'tqdm',
   ],
)
