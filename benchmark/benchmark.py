#!/usr/bin/env python
#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

"""MIPLearn Benchmark Scripts

Usage:
    benchmark.py train [options] <challenge>
    benchmark.py test-baseline [options] <challenge>
    benchmark.py test-ml [options] <challenge>
    benchmark.py charts <challenge>
    
Options:
    -h --help               Show this screen
    --jobs=<n>              Number of instances to solve simultaneously [default: 10]
    --train-time-limit=<n>  Solver time limit during training in seconds [default: 3600]
    --test-time-limit=<n>   Solver time limit during test in seconds [default: 900]
    --solver-threads=<n>    Number of threads the solver is allowed to use [default: 4]
    --solver=<s>            Internal MILP solver to use [default: gurobi]
"""
import importlib
import logging
import pathlib
import pickle
import sys

from docopt import docopt
from numpy import median

from miplearn import LearningSolver, BenchmarkRunner

logging.basicConfig(
    format="%(asctime)s %(levelname).1s %(name)s: %(message)12s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logging.getLogger("gurobipy").setLevel(logging.ERROR)
logging.getLogger("pyomo.core").setLevel(logging.ERROR)
logging.getLogger("miplearn").setLevel(logging.INFO)
logger = logging.getLogger("benchmark")

args = docopt(__doc__)
basepath = args["<challenge>"]
pathlib.Path(basepath).mkdir(parents=True, exist_ok=True)

n_jobs = int(args["--jobs"])
n_threads = int(args["--solver-threads"])
train_time_limit = int(args["--train-time-limit"])
test_time_limit = int(args["--test-time-limit"])
internal_solver = args["--solver"]


def save(obj, filename):
    logger.info("Writing %s..." % filename)
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def load(filename):
    import pickle

    with open(filename, "rb") as file:
        return pickle.load(file)


def train():
    problem_name, challenge_name = args["<challenge>"].split("/")
    pkg = importlib.import_module("miplearn.problems.%s" % problem_name)
    challenge = getattr(pkg, challenge_name)()
    train_instances = challenge.training_instances
    test_instances = challenge.test_instances
    solver = LearningSolver(
        time_limit=train_time_limit,
        solver=internal_solver,
        threads=n_threads,
    )
    solver.parallel_solve(train_instances, n_jobs=n_jobs)
    save(train_instances, "%s/train_instances.bin" % basepath)
    save(test_instances, "%s/test_instances.bin" % basepath)


def test_baseline():
    test_instances = load("%s/test_instances.bin" % basepath)
    solvers = {
        "baseline": LearningSolver(
            time_limit=test_time_limit,
            solver=internal_solver,
            threads=n_threads,
        ),
    }
    benchmark = BenchmarkRunner(solvers)
    benchmark.parallel_solve(test_instances, n_jobs=n_jobs)
    benchmark.save_results("%s/benchmark_baseline.csv" % basepath)


def test_ml():
    logger.info("Loading instances...")
    train_instances = load("%s/train_instances.bin" % basepath)
    test_instances = load("%s/test_instances.bin" % basepath)
    solvers = {
        "ml-exact": LearningSolver(
            time_limit=test_time_limit,
            solver=internal_solver,
            threads=n_threads,
        ),
        "ml-heuristic": LearningSolver(
            time_limit=test_time_limit,
            solver=internal_solver,
            threads=n_threads,
            mode="heuristic",
        ),
    }
    benchmark = BenchmarkRunner(solvers)
    logger.info("Loading results...")
    benchmark.load_results("%s/benchmark_baseline.csv" % basepath)
    logger.info("Fitting...")
    benchmark.fit(train_instances)
    logger.info("Solving...")
    benchmark.parallel_solve(test_instances, n_jobs=n_jobs)
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

    sense = results.loc[0, "Sense"]
    if sense == "min":
        primal_column = "Relative Upper Bound"
        obj_column = "Upper Bound"
        predicted_obj_column = "Predicted UB"
    else:
        primal_column = "Relative Lower Bound"
        obj_column = "Lower Bound"
        predicted_obj_column = "Predicted LB"

    palette = {"baseline": "#9b59b6", "ml-exact": "#3498db", "ml-heuristic": "#95a5a6"}
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(12, 4),
        gridspec_kw={"width_ratios": [2, 1, 1, 2]},
    )
    sns.stripplot(
        x="Solver",
        y="Wallclock Time",
        data=results,
        ax=ax1,
        jitter=0.25,
        palette=palette,
        size=4.0,
    )
    sns.barplot(
        x="Solver",
        y="Wallclock Time",
        data=results,
        ax=ax1,
        errwidth=0.0,
        alpha=0.4,
        palette=palette,
        estimator=median,
    )
    ax1.set(ylabel="Wallclock Time (s)")
    ax2.set_ylim(-0.5, 5.5)
    sns.stripplot(
        x="Solver",
        y="Gap (%)",
        jitter=0.25,
        data=results[results["Solver"] != "ml-heuristic"],
        ax=ax2,
        palette=palette,
        size=4.0,
    )
    ax3.set_ylim(0.95, 1.05)
    sns.stripplot(
        x="Solver",
        y=primal_column,
        jitter=0.25,
        data=results[results["Solver"] == "ml-heuristic"],
        ax=ax3,
        palette=palette,
    )

    sns.scatterplot(
        x=obj_column,
        y=predicted_obj_column,
        hue="Solver",
        data=results[results["Solver"] == "ml-exact"],
        ax=ax4,
        palette=palette,
    )
    xlim, ylim = ax4.get_xlim(), ax4.get_ylim()
    ax4.plot([-1e10, 1e10], [-1e10, 1e10], ls="-", color="#cccccc")
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.get_legend().remove()

    fig.tight_layout()
    plt.savefig("%s/performance.png" % basepath, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    if args["train"]:
        train()
    if args["test-baseline"]:
        test_baseline()
    if args["test-ml"]:
        test_ml()
    if args["charts"]:
        charts()
