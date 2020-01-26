MIPLearn
========

**MIPLearn** is a flexible and extensible framework for *Learning-Enhanced Mixed-Integer Optimization*. It was designed to efficiently handle discrete optimization problems that need to be repeatedly solved with only relatively minor changes to the input data. The package uses Machine Learning (ML) to automatically identify patterns in previously solved instances of the problem, or in the solution process itself, and produces hints that can guide a traditional MIP solver towards the optimal solution faster. For particular classes of problems, this approach has been shown to provide significant performance benefits (see references below).

Table of contents
-----------------
  * [Features](#features)
  * [Installation](#installation)
  * [Basic Usage](#basic-usage)
     * [Using LearningSolver](#using-learningsolver)
     * [Describing problem instances](#describing-problem-instances)
     * [Obtaining heuristic solutions](#obtaining-heuristic-solutions)
     * [Saving and loading solver state](#saving-and-loading-solver-state)
     * [Solving training instances in parallel](#solving-training-instances-in-parallel)
  * [Benchmarking](#benchmarking)
     * [Using BenchmarkRunner](#using-benchmarkrunner)
     * [Saving and loading benchmark results](#saving-and-loading-benchmark-results)
  * [Customization](#customization)
     * [Selecting the internal MIP solver](#selecting-the-internal-mip-solver)
  * [Current Limitations](#current-limitations)
  * [References](#references)
  * [Authors](#authors)
  * [License](#license)

Features
--------
* **MIPLearn proposes a flexible, problem-agnostic way** for users to describe optimization problems to a Learning-Enhanced Solver, from both the MIP perspective and from the ML perspective. MIP formulations are specified as [Pyomo](https://www.pyomo.org/) models, while features describing instances and decision variables are specified as [NumPy](https://numpy.org/) arrays. Users can easily experiment with different mathematical formulations and ML encodings.

* **MIPLearn provides a reference implementation of a *Learning-Enhanced Solver*,** which can use the above problem specification to automatically predict, based on previously solved instances: (i) partial solutions which are likely to work well as MIP starts, (ii) an initial set of lazy constraints to enforce and (iii) affine subspaces where the solution is likely to reside. This process is entirely transparent to the user: the most suitable ML models are automatically selected, trained and cross-validated with no user intervention.

* **MIPLearn is customizable and extensible**. For MIP and ML researchers exploring new techniques to accelerate MIP performance based on historical data, each component of the reference solver can be individually replaced or customized.

* **MIPLearn provides a set of benchmark problems and random instance generators,** covering applications from different domains, which can be used to quickly evaluate new learning-enhanced MIP techniques in a measurable and reproducible way.

Installation
------------

The package is currently only available for Python and Pyomo. It can be installed using `pip` as follows:

```bash
pip install git+ssh://git@github.com/iSoron/miplearn.git
```

Basic Usage
-----------

### Using `LearningSolver`

The main class provided by this package is `LearningSolver`, a reference learning-enhanced MIP solver which automatically extracts information from previous runs to accelerate the solution of new instances. Assuming we already have a list of instances to solve, `LearningSolver` can be used as follows:

```python
from miplearn import LearningSolver

all_instances = ... # user-provided list of instances to solve
solver = LearningSolver()
for instance in all_instances:
    solver.solve(instance)
    solver.fit()
```

During the first call to `solver.solve(instance)`, the solver will process the instance from scratch, since no historical information is available, but it will already start gathering information. By calling `solver.fit()`, we instruct the solver to train all the internal Machine Learning models based on the information gathered so far. As this operation can be expensive, it may  be performed after a larger batch of instances has been solved, instead of after every solve. After the first call to `solver.fit()`, subsequent calls to `solver.solve(instance)` will automatically use the trained Machine Learning models to accelerate the solution process.

### Describing problem instances

Instances to be solved by `LearningSolver` must derive from the abstract class `miplearn.Instance`. The following three abstract methods must be implemented:

* `instance.to_model()`, which returns a concrete Pyomo model corresponding to the instance;
* `instance.get_instance_features()`, which returns a 1-dimensional Numpy array of (numerical) features describing the entire instance;
* `instance.get_variable_features(var, index)`, which returns a 1-dimensional array of (numerical) features describing a particular decision variable.


The first method is used by `LearningSolver` to construct a concrete Pyomo model, which will be provided to the internal MIP solver. The user should keep a reference to this Pyomo model, in order to retrieve, for example, the optimal variable values.

The second and third methods provide an encoding of the instance, which can be used by the ML models to make predictions. In the knapsack problem, for example, an implementation may decide to provide as instance features the average weights, average prices, number of items and the size of the knapsack. The weight and the price of each individual item could be provided as variable features. See `miplearn/problems/knapsack.py` for a concrete example.

An optional method which can be implemented is `instance.get_variable_category(var, index)`, which returns a category (a string, an integer or any hashable type) for each decision variable. If two variables have the same category, `LearningSolver` will use the same internal ML model to predict the values of both variables. By default, all variables belong to the `"default"` category, and therefore only one ML model is used for all variables.

It is not necessary to have a one-to-one correspondence between features and problem instances. One important (and deliberate) limitation of MIPLearn, however, is that `get_instance_features()` must always return arrays of same length for all relevant instances of the problem. Similarly, `get_variable_features(var, index)` must also always return arrays of same length for all variables in each category. It is up to the user to decide how to encode variable-length characteristics of the problem into fixed-length vectors. In graph problems, for example, graph embeddings can be used to reduce the (variable-length) lists of nodes and edges into a fixed-length structure that still preserves some properties of the graph. Different instance encodings may have significant impact on performance.

### Obtaining heuristic solutions

By default, `LearningSolver` uses Machine Learning to accelerate the MIP solution process, but keeps all optimality guarantees typically provided by MIP solvers. In the default mode of operation, predicted optimal solutions, for example, are used only as MIP starts.

For more significant performance benefits, `LearningSolver` can also be configured to place additional trust in the Machine Learning predictors, by using the `mode="heuristic"` constructor argument. When operating in this mode, if a ML model is statistically shown (through *stratified k-fold cross validation*) to have exceptionally high accuracy, the solver may decide to restrict the search space based on its predictions. Parts of the solution which the ML models cannot predict accurately will still be explored using traditional (branch-and-bound) methods. This mode naturally loses all optimality guarantees, but, for particular applications, it has been shown to quickly produce optimal or near-optimal solutions (see references below).

**Note:** *The heuristic mode should only be used if the solver is first trained on a large and statistically representative set of training instances.*

### Saving and loading solver state

After solving a large number of training instances, it may be desirable to save the current state of `LearningSolver` to disk, so that the solver can still use the acquired knowledge after the application restarts. This can be accomplished by using the methods `solver.save(filename)` and `solver.load(filename)`, as the following example illustrates:

```python
from miplearn import LearningSolver

solver = LearningSolver()
for instance in some_instances:
    solver.solve(instance)
solver.fit()
solver.save("/tmp/miplearn.bin")

# Application restarts...

solver = LearningSolver()
solver.load("/tmp/miplearn.bin")
for instance in more_instances:
    solver.solve(instance)
```

In addition to storing the training data, `solver.save` also serializes and stores all trained ML models themselves, so it is not necessary to call `solver.fit`.


### Solving training instances in parallel

In many situations, training instances can be solved in parallel to accelerate the training process. `LearningSolver` provides the method `parallel_solve(instances)` to easily achieve this. After all instances have been solved, the ML models can be trained and saved to disk as usual, as the next example illustrates:

```python
from miplearn import LearningSolver

# Training phase...
solver = LearningSolver(...) # training solver parameters
solver.parallel_solve(training_instances, n_jobs=4)
solver.fit()
solver.save("/tmp/data.bin")

# Test phase...
solver = LearningSolver(...) # test solver parameters
solver.load("/tmp/data.bin")
solver.solve(test_instance)
```

Benchmarking
------------

### Using `BenchmarkRunner`

MIPLearn provides the utility class `BenchmarkRunner`, which simplifies the task of comparing the performance of different solvers. The snippet below shows its basic usage:

```python
from miplearn import BenchmarkRunner, LearningSolver

# Create train and test instances
train_instances = [...]
test_instances  = [...]

# Training phase...
training_solver = LearningSolver(...)
training_solver.parallel_solve(train_instances, n_jobs=10)
training_solver.save("data.bin")

# Test phase...
test_solvers = {
    "Baseline": LearningSolver(...), # each solver may have different parameters
    "Strategy A": LearningSolver(...), 
    "Strategy B": LearningSolver(...),
    "Strategy C": LearningSolver(...),
}
benchmark = BenchmarkRunner(test_solvers)
benchmark.load_fit("data.bin")
benchmark.parallel_solve(test_instances, n_jobs=2)
print(benchmark.raw_results())
```

The method `load_fit` loads the saved training data into each one of the provided solvers and trains their respective ML models. The method `parallel_solve` solves the test instances in parallel, and collects solver statistics such as running time and optimal value. Finally, `raw_results` produces a table of results (Pandas DataFrame) with the following columns:

* **Solver,** the name of the solver.
* **Instance,** the sequence number identifying the instance.
* **Wallclock Time,** the wallclock running time (in seconds) spent by the solver;
* **Obj Value,** the objective value of the solution found by the solver;
* **Relative Wallclock Time,** a number indicating how many times slower this run was when compared to the best time achieved by any solver when processing this instance. For example, if this run took 10 seconds, but another solver took only 5 seconds to solve the same instance, the relative wallclock time would be 2.00.
* **Relative Obj Value,** how many times better (or worse) this solution was in terms of objective value, when compared to the solutions produced by the other solvers for the same instance. For example, if this solver found a solution with objective value 100.0 on a minimization problem, and another solver found a solution with value 80.0, then the relative objective value would be 1.25.

### Saving and loading benchmark results

When iteratively exploring new formulations, encoding and solver parameters, it is often desirable to avoid repeating parts of the benchmark suite. For example, if the baseline solver has not been changed, there is no need to evaluate its performance again and again when making small changes to the remaining solvers. `BenchmarkRunner` provides the methods `save` and `load`, which can be used to avoid this repetition, as the next example shows:

```python
# Benchmark baseline solvers and save results to a file.
benchmark = BenchmarkRunner(baseline_solvers)
benchmark.load_fit("training_data.bin")
benchmark.parallel_solve(test_instances)
benchmark.save_results("baseline_results.csv")

# Benchmark remaining solvers, loading baseline results from file.
benchmark = BenchmarkRunner(alternative_solvers)
benchmark.load_results("baseline_results.csv")
benchmark.load_fit("training_data.bin")
benchmark.parallel_solve(test_instances)
```

Customization
-------------

### Selecting the internal MIP solver

By default, `LearningSolver` uses [Gurobi](https://www.gurobi.com/) as its internal MIP solver. Alternative solvers can be specified through the `internal_solver_factory` constructor argument. This argument should provide a function (with no arguments) that constructs, configures and returns the desired solver. To select CPLEX, for example:
```python
from miplearn import LearningSolver
import pyomo.environ as pe

def cplex_factory():
    cplex = pe.SolverFactory("cplex_persistent")
    cplex.options["threads"] = 4
    return cplex

solver = LearningSolver(internal_solver_factory=cplex_factory)
```


Current Limitations
-------------------

* Only binary and continuous decision variables are currently supported.
* Solver callbacks (lazy constraints, cutting planes) are not currently supported.

References
----------

* **Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems.** *Alinson S. Xavier, Feng Qiu, Shabbir Ahmed*. INFORMS Journal on Computing (to appear). https://arxiv.org/abs/1902.01697

Authors
-------
* **Alinson S. Xavier,** Argonne National Laboratory <<axavier@anl.gov>>

License
-------

    MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
    Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
