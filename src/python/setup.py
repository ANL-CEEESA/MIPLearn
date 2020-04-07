from setuptools import setup

setup(
    name='miplearn',
    version='0.1',
    description='A Machine-Learning Framework for Mixed-Integer Optimization',
    author='Alinson S. Xavier',
    author_email='axavier@anl.gov',
    packages=['miplearn'],
    install_requires=[
        'docopt',
        'matplotlib',
        'mkdocs',
        'mkdocs-cinder',
        'networkx',
        'numpy',
        'pandas',
        'p_tqdm',
        'pyomo',
        'pytest',
        'pytest-watch',
        'python-markdown-math',
        'seaborn',
        'sklearn',
        'tqdm',
   ],
)
