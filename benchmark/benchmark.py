#!/usr/bin/env python
"""Benchmark script

Usage:
    benchmark.py train <challenge>
    benchmark.py test-baseline <challenge>
    benchmark.py test-ml <challenge>
    benchmark.py charts <challenge>
    
Options:
    -h --help    Show this screen
"""
from docopt import docopt
import importlib, pathlib
from miplearn import (LearningSolver,
                      BenchmarkRunner,
                      WarmStartComponent,
                      BranchPriorityComponent,
                     )
from numpy import median
import pyomo.environ as pe
import pickle

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

args = docopt(__doc__)
basepath = args["<challenge>"]
pathlib.Path(basepath).mkdir(parents=True, exist_ok=True)


def save(obj, filename):
    print("Writing %s..." % filename)
    with open(filename, "wb") as file:
        pickle.dump(obj, file)
        
        
def load(filename):
    import pickle
    with open(filename, "rb") as file:
        return pickle.load(file)        
        
        
def train_solver_factory():
    solver = pe.SolverFactory('gurobi_persistent')
    solver.options["threads"] = 4
    solver.options["TimeLimit"] = 300
    return solver


def test_solver_factory():
    solver = pe.SolverFactory('gurobi_persistent')
    solver.options["threads"] = 4
    solver.options["TimeLimit"] = 300
    return solver


def train():
    problem_name, challenge_name = args["<challenge>"].split("/")
    pkg = importlib.import_module("miplearn.problems.%s" % problem_name)
    challenge = getattr(pkg, challenge_name)()
    train_instances = challenge.training_instances
    test_instances  = challenge.test_instances
    solver = LearningSolver(
        internal_solver_factory=train_solver_factory,
        components={
            "warm-start": WarmStartComponent(),
            #"branch-priority": BranchPriorityComponent(),
        },
    )
    solver.parallel_solve(train_instances, n_jobs=10)
    solver.fit(n_jobs=10)
    solver.save_state("%s/training_data.bin" % basepath)
    save(train_instances, "%s/train_instances.bin" % basepath)
    save(test_instances, "%s/test_instances.bin" % basepath)
    
    
def test_baseline():
    solvers = {
        "baseline": LearningSolver(
            internal_solver_factory=test_solver_factory,
            components={},
        ),
    }
    test_instances = load("%s/test_instances.bin" % basepath)
    benchmark = BenchmarkRunner(solvers)
    benchmark.parallel_solve(test_instances, n_jobs=10)
    benchmark.save_results("%s/benchmark_baseline.csv" % basepath)
    
    
def test_ml():
    solvers = {
        "ml-exact": LearningSolver(
            internal_solver_factory=test_solver_factory,
            components={
                "warm-start": WarmStartComponent(),
                #"branch-priority": BranchPriorityComponent(),
            },
        ),
        "ml-heuristic": LearningSolver(
            internal_solver_factory=test_solver_factory,
            mode="heuristic",
            components={
                "warm-start": WarmStartComponent(),
                #"branch-priority": BranchPriorityComponent(),
            },
        ),
    }
    test_instances = load("%s/test_instances.bin" % basepath)
    benchmark = BenchmarkRunner(solvers)
    benchmark.load_state("%s/training_data.bin" % basepath)
    benchmark.load_results("%s/benchmark_baseline.csv" % basepath)
    benchmark.parallel_solve(test_instances, n_jobs=10)
    benchmark.save_results("%s/benchmark_ml.csv" % basepath)    
    
    
def charts():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("Blues_r")
    benchmark = BenchmarkRunner({})
    benchmark.load_results("%s/benchmark_ml.csv" % basepath)
    results = benchmark.raw_results()
    results["Gap (%)"] = results["Gap"] * 100.0
    palette={
        "baseline": "#9b59b6", 
        "ml-exact": "#3498db",
        "ml-heuristic": "#95a5a6"
    }
    fig, axes = plt.subplots(nrows=1,
                             ncols=3,
                             figsize=(10,4),
                             gridspec_kw={'width_ratios': [3, 3, 2]},
                            )
    sns.stripplot(x="Solver",
                  y="Wallclock Time",
                  data=results,
                  ax=axes[0],
                  jitter=0.25,
                  palette=palette,
               );
    sns.barplot(x="Solver",
                y="Wallclock Time",
                data=results,
                ax=axes[0],
                errwidth=0.,
                alpha=0.3,
                palette=palette,
                estimator=median,
               );
    axes[0].set(ylabel='Wallclock Time (s)')
    axes[1].set_ylim(-0.5, 5.5)
    sns.stripplot(x="Solver",
                  y="Gap (%)",
                  jitter=0.25,
                  data=results[results["Solver"] != "ml-heuristic"],
                  ax=axes[1],
                  palette=palette,
                 );
    axes[2].set_ylim(0.95,1.01)
    sns.stripplot(x="Solver",
                  y="Relative Lower Bound",
                  jitter=0.25,
                  data=results[results["Solver"] == "ml-heuristic"],
                  ax=axes[2],
                  palette=palette,
                 );
    fig.tight_layout()
    plt.savefig("%s/performance.png" % basepath,
                bbox_inches='tight',
                dpi=150)

if __name__ == "__main__":
    if args["train"]:
        train()
    if args["test-baseline"]:
        test_baseline()
    if args["test-ml"]:
        test_ml()
    if args["charts"]:
        charts()