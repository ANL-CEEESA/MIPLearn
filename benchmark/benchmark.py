#!/usr/bin/env python
#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

"""MIPLearn Benchmark Scripts

Usage:
    benchmark.py train [options] <challenge>
    benchmark.py test-baseline [options] <challenge>
    benchmark.py test-ml [options] <challenge>
    benchmark.py charts <challenge>
    
Options:
    -h --help               Show this screen
    --train-jobs=<n>        Number of instances to solve in parallel during training [default: 10]
    --train-time-limit=<n>  Solver time limit during training in seconds [default: 3600]
    --test-jobs=<n>         Number of instances to solve in parallel during test [default: 5]
    --test-time-limit=<n>   Solver time limit during test in seconds [default: 900]
    --solver-threads=<n>    Number of threads the solver is allowed to use [default: 4]
"""
import glob
import importlib
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from docopt import docopt
from numpy import median

from miplearn import (
    LearningSolver,
    BenchmarkRunner,
    GurobiPyomoSolver,
    setup_logger,
    PickleGzInstance,
    write_pickle_gz_multiple,
)

setup_logger()
logging.getLogger("gurobipy").setLevel(logging.ERROR)
logging.getLogger("pyomo.core").setLevel(logging.ERROR)
logger = logging.getLogger("benchmark")


def train(args):
    basepath = args["<challenge>"]
    problem_name, challenge_name = args["<challenge>"].split("/")
    pkg = importlib.import_module(f"miplearn.problems.{problem_name}")
    challenge = getattr(pkg, challenge_name)()

    if not os.path.isdir(f"{basepath}/train"):
        write_pickle_gz_multiple(challenge.training_instances, f"{basepath}/train")
        write_pickle_gz_multiple(challenge.test_instances, f"{basepath}/test")

    done_filename = f"{basepath}/train/done"
    if not os.path.isfile(done_filename):
        train_instances = [
            PickleGzInstance(f) for f in glob.glob(f"{basepath}/train/*.gz")
        ]
        solver = LearningSolver(
            solver=lambda: GurobiPyomoSolver(
                params={
                    "TimeLimit": int(args["--train-time-limit"]),
                    "Threads": int(args["--solver-threads"]),
                }
            ),
        )
        solver.parallel_solve(
            train_instances,
            n_jobs=int(args["--train-jobs"]),
        )
        Path(done_filename).touch(exist_ok=True)


def test_baseline(args):
    basepath = args["<challenge>"]
    test_instances = [PickleGzInstance(f) for f in glob.glob(f"{basepath}/test/*.gz")]
    csv_filename = f"{basepath}/benchmark_baseline.csv"
    if not os.path.isfile(csv_filename):
        solvers = {
            "baseline": LearningSolver(
                solver=lambda: GurobiPyomoSolver(
                    params={
                        "TimeLimit": int(args["--test-time-limit"]),
                        "Threads": int(args["--solver-threads"]),
                    }
                ),
            ),
        }
        benchmark = BenchmarkRunner(solvers)
        benchmark.parallel_solve(
            test_instances,
            n_jobs=int(args["--test-jobs"]),
        )
        benchmark.write_csv(csv_filename)


def test_ml(args):
    basepath = args["<challenge>"]
    test_instances = [PickleGzInstance(f) for f in glob.glob(f"{basepath}/test/*.gz")]
    train_instances = [PickleGzInstance(f) for f in glob.glob(f"{basepath}/train/*.gz")]
    csv_filename = f"{basepath}/benchmark_ml.csv"
    if not os.path.isfile(csv_filename):
        solvers = {
            "ml-exact": LearningSolver(
                solver=lambda: GurobiPyomoSolver(
                    params={
                        "TimeLimit": int(args["--test-time-limit"]),
                        "Threads": int(args["--solver-threads"]),
                    }
                ),
            ),
            "ml-heuristic": LearningSolver(
                solver=lambda: GurobiPyomoSolver(
                    params={
                        "TimeLimit": int(args["--test-time-limit"]),
                        "Threads": int(args["--solver-threads"]),
                    }
                ),
                mode="heuristic",
            ),
        }
        benchmark = BenchmarkRunner(solvers)
        benchmark.fit(train_instances)
        benchmark.parallel_solve(
            test_instances,
            n_jobs=int(args["--test-jobs"]),
        )
        benchmark.write_csv(csv_filename)


def charts(args):
    basepath = args["<challenge>"]
    sns.set_style("whitegrid")
    sns.set_palette("Blues_r")

    csv_files = [
        f"{basepath}/benchmark_baseline.csv",
        f"{basepath}/benchmark_ml.csv",
    ]
    results = pd.concat(map(pd.read_csv, csv_files))
    groups = results.groupby("Instance")
    best_lower_bound = groups["Lower bound"].transform("max")
    best_upper_bound = groups["Upper bound"].transform("min")
    results["Relative lower bound"] = results["Lower bound"] / best_lower_bound
    results["Relative upper bound"] = results["Upper bound"] / best_upper_bound

    sense = results.loc[0, "Sense"]
    if (sense == "min").any():
        primal_column = "Relative upper bound"
        obj_column = "Upper bound"
        predicted_obj_column = "Objective: Predicted upper bound"
    else:
        primal_column = "Relative lower bound"
        obj_column = "Lower bound"
        predicted_obj_column = "Objective: Predicted lower bound"

    palette = {"baseline": "#9b59b6", "ml-exact": "#3498db", "ml-heuristic": "#95a5a6"}
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(12, 4),
        gridspec_kw={"width_ratios": [2, 1, 1, 2]},
    )

    # Wallclock time
    sns.stripplot(
        x="Solver",
        y="Wallclock time",
        data=results,
        ax=ax1,
        jitter=0.25,
        palette=palette,
        size=4.0,
    )
    sns.barplot(
        x="Solver",
        y="Wallclock time",
        data=results,
        ax=ax1,
        errwidth=0.0,
        alpha=0.4,
        palette=palette,
        estimator=median,
    )
    ax1.set(ylabel="Wallclock time (s)")

    # Gap
    ax2.set_ylim(-0.5, 5.5)
    sns.stripplot(
        x="Solver",
        y="Gap",
        jitter=0.25,
        data=results[results["Solver"] != "ml-heuristic"],
        ax=ax2,
        palette=palette,
        size=4.0,
    )

    # Relative primal bound
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

    # Predicted vs actual primal bound
    xlim, ylim = ax4.get_xlim(), ax4.get_ylim()
    ax4.plot(
        [-1e10, 1e10],
        [-1e10, 1e10],
        ls="-",
        color="#cccccc",
    )
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.get_legend().remove()
    ax4.set(
        ylabel="Predicted value",
        xlabel="Actual value",
    )

    fig.tight_layout()
    plt.savefig(
        f"{basepath}/performance.png",
        bbox_inches="tight",
        dpi=150,
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    if args["train"]:
        train(args)
    if args["test-baseline"]:
        test_baseline(args)
    if args["test-ml"]:
        test_ml(args)
    if args["charts"]:
        charts(args)
