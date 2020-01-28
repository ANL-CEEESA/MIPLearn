MIPLearn
========

**MIPLearn** is a flexible and extensible framework for *Learning-Enhanced Mixed-Integer Optimization*, an approach aimed at efficiently handling challenging discrete optimization problems that need to be repeatedly solved with only relatively minor changes to the input data. The package uses Machine Learning (ML) to automatically identify patterns in previously solved instances of the problem, or in the solution process itself, and produces hints that can guide a traditional MIP solver towards the optimal solution faster. For particular classes of problems, this approach has been shown to provide significant performance benefits (see references below).

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
* **MIPLearn proposes a flexible problem specification format,** which allows users to describe their particular optimization problems to a Learning-Enhanced MIP solver, both from the MIP perspective and from the ML perspective, without making any assumptions on the problem being modeled, the mathematical formulation of the problem, or ML encoding. While the format is very flexible, some constraints are enforced to ensure that it is usable by an actual solver.

* **MIPLearn provides a reference implementation of a *Learning-Enhanced Solver*,** which can use the above problem specification format to automatically predict, based on previously solved instances, a number of hints to accelerate MIP performance. Currently, the reference solver is able to predict: (i) partial solutions which are likely to work well as MIP starts; (ii) an initial set of lazy constraints to enforce; (iii) affine subspaces where the solution is likely to reside; (iv) variable branching priorities to accelerate the exploration of the branch-and-bound tree. The usage of the solver is very straightforward. The most suitable ML models are automatically selected, trained, cross-validated and applied to the problem with no user intervention.

* **MIPLearn provides a set of benchmark problems and random instance generators,** covering applications from different domains, which can be used to quickly evaluate new learning-enhanced MIP techniques in a measurable and reproducible way.

* **MIPLearn is customizable and extensible**. For MIP and ML researchers exploring new techniques to accelerate MIP performance based on historical data, each component of the reference solver can be individually replaced, extended or customized.

Installation
------------

The package is currently available for Python and Pyomo. It can be installed using `pip` as follows:

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

By default, `LearningSolver` uses Machine Learning to accelerate the MIP solution process, while maintaining all optimality guarantees provided by the MIP solver. In the default mode of operation, for example, predicted optimal solutions are used only as MIP starts.

For more significant performance benefits, `LearningSolver` can also be configured to place additional trust in the Machine Learning predictors, by using the `mode="heuristic"` constructor argument. When operating in this mode, if a ML model is statistically shown (through *stratified k-fold cross validation*) to have exceptionally high accuracy, the solver may decide to restrict the search space based on its predictions. The parts of the solution which the ML models cannot predict accurately will still be explored using traditional (branch-and-bound) methods

This mode naturally loses all optimality guarantees, and therefore should only be used if the solver is first trained on a large and representative set of training instances. For particular applications, however, this mode has been shown to quickly produce optimal or near-optimal solutions (see references below).

### Saving and loading solver state

After solving a large number of training instances, it may be desirable to save the current state of `LearningSolver` to disk, so that the solver can still use the acquired knowledge after the application restarts. This can be accomplished by using the methods `solver.save_state(filename)` and `solver.load_state(filename)`, as the following example illustrates:

```python
from miplearn import LearningSolver

solver = LearningSolver()
for instance in some_instances:
    solver.solve(instance)
solver.fit()
solver.save_state("/tmp/state.bin")

# Application restarts...

solver = LearningSolver()
solver.load_state("/tmp/state.bin")
for instance in more_instances:
    solver.solve(instance)
```

In addition to storing the training data, `save_state` also stores all trained ML models. Therefore, if the the models were trained before saving the state to disk, it is not necessary to train them again after loading.


### Solving training instances in parallel

In many situations, training instances can be solved in parallel to accelerate the training process. `LearningSolver` provides the method `parallel_solve(instances)` to easily achieve this:

```python
from miplearn import LearningSolver

# Training phase...
solver = LearningSolver(...) # training solver parameters
solver.parallel_solve(training_instances, n_jobs=4)
solver.fit()
solver.save_state("/tmp/data.bin")

# Test phase...
solver = LearningSolver(...) # test solver parameters
solver.load_state("/tmp/data.bin")
solver.solve(test_instance)
```

After all training instances have been solved in parallel, the ML models can be trained and saved to disk as usual, using `fit` and `save_state`, as explained in the previous subsections.

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
training_solver.save_state("data.bin")

# Test phase...
test_solvers = {
    "Baseline": LearningSolver(...), # each solver may have different parameters
    "Strategy A": LearningSolver(...), 
    "Strategy B": LearningSolver(...),
    "Strategy C": LearningSolver(...),
}
benchmark = BenchmarkRunner(test_solvers)
benchmark.load_state("data.bin")
benchmark.fit()
benchmark.parallel_solve(test_instances, n_jobs=2)
print(benchmark.raw_results())
```

The method `load_state` loads the saved training data into each one of the provided solvers, while `fit` trains their respective ML models. The method `parallel_solve` solves the test instances in parallel, and collects solver statistics such as running time and optimal value. Finally, `raw_results` produces a table of results (Pandas DataFrame) with the following columns:

* **Solver,** the name of the solver.
* **Instance,** the sequence number identifying the instance.
* **Wallclock Time,** the wallclock running time (in seconds) spent by the solver;
* **Lower Bound,** the best lower bound obtained by the solver;
* **Upper Bound,** the best upper bound obtained by the solver;
* **Gap,** the relative MIP integrality gap at the end of the optimization;
* **Nodes,** the number of explorer branch-and-bound nodes.

In addition to the above, there is also a *Relative* version of most columns, where the raw number is compared to the solver which provided the best performance. The *Relative Wallclock Time* for example, indicates how many times slower this run was when compared to the best time achieved by any solver when processing this instance. For example, if this run took 10 seconds, but the fastest solver took only 5 seconds to solve the same instance, the relative wallclock time would be 2.

### Saving and loading benchmark results

When iteratively exploring new formulations, encoding and solver parameters, it is often desirable to avoid repeating parts of the benchmark suite. For example, if the baseline solver has not been changed, there is no need to evaluate its performance again and again when making small changes to the remaining solvers. `BenchmarkRunner` provides the methods `save_results` and `load_results`, which can be used to avoid this repetition, as the next example shows:

```python
# Benchmark baseline solvers and save results to a file.
benchmark = BenchmarkRunner(baseline_solvers)
benchmark.load_state("training_data.bin")
benchmark.parallel_solve(test_instances)
benchmark.save_results("baseline_results.csv")

# Benchmark remaining solvers, loading baseline results from file.
benchmark = BenchmarkRunner(alternative_solvers)
benchmark.load_state("training_data.bin")
benchmark.load_results("baseline_results.csv")
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
    cplex = pe.SolverFactory("cplex")
    cplex.options["threads"] = 4
    return cplex

solver = LearningSolver(internal_solver_factory=cplex_factory)
```


Current Limitations
-------------------

* Only binary and continuous decision variables are currently supported.
* Solver callbacks (lazy constraints, cutting planes) are not currently supported.
* Only `gurobi_persistent` is currently fully supported by all components. Other solvers may work if some components are disabled.

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
