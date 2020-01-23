MIPLearn
========

**MIPLearn** is a flexible and extensible framework for *Learning-Enhanced Mixed-Integer Optimization*. It was designed to efficiently handle discrete optimization problems that need to be repeatedly solved with only relatively minor changes to the input data. The package uses Machine Learning (ML) to automatically identify patterns in previously solved instances of the problem, or in the solution process itself, and produces hints that can guide a traditional MIP solver, such as CPLEX and Gurobi, towards the optimal solution faster. For particular classes of problems, this approach has been shown to provide significant performance benefits (see references below).

Features
--------
* **MIPLearn proposes a flexible, problem-agnostic way** for users to describe optimization problems to a Learning-Enhanced Solver, from both the MIP perspective and from the ML perspective. MIP formulations are specified as [Pyomo](https://www.pyomo.org/) models, while features describing instances and decision variables are specified as [NumPy](https://numpy.org/) arrays. Users can easily experiment with different mathematical formulations and ML encodings.

* **MIPLearn provides a reference implementation of a *Learning-Enhanced Solver*,** which can use the above problem specification to automatically predict, based on previously solved instances: (i) partial solutions which are likely to work well as MIP starts, (ii) an initial set of lazy constraints to enforce and (iii) affine subspaces where the solution is likely to reside. This process is entirely transparent to the user: the most suitable ML models are automatically selected, trained and cross-validated with no user intervention.

* **MIPLearn is customizable and extensible**. For MIP and ML researchers exploring new techniques to accelerate MIP performance based on historical data, each component of the reference solver can be individually replaced or customized. The reference solver can use many open-source and commercial MIP solvers as backend, including Cbc, SCIP, CPLEX and Gurobi.

* **MIPLearn provides a set of benchmark problems and random instance generators,** covering applications from different domains, which can be used to quickly evaluate new learning-enhanced MIP techniques in a measurable and reproducible way.

Installation
------------

The package is currently only available for Python and Pyomo. It can be installed using `pip` as follows:

```bash
pip install git+ssh://git@github.com/iSoron/miplearn.git
```

Usage
-----

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

Selecting the internal MIP solver
---------------------------------
By default, `LearningSolver` uses Cbc as its internal MIP solver. Alternative solvers can be specified through the `parent_solver`a argument, as follows. Persistent Pyomo solvers are supported. To select Gurobi, for example:
```python
from miplearn import LearningSolver
import pyomo.environ as pe

gurobi = pe.SolverFactory("gurobi_persistent")
solver = LearningSolver(parent_solver=gurobi)
```

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

For more signifcant performance benefits, `LearningSolver` can also be configured to place additional trust in the Machine Learning predictors, using the `mode="heuristic"` constructor argument. When operating in this mode, if a ML model is statistically shown (through stratified k-fold cross validation) to have exceptionally high accuracy, the solver may decide to restrict the search space based on its predictions. Parts of the solution which the ML models cannot predict accurately will still be explored using traditional (branch-and-bound) methods. This mode naturally loses all optimality guarantees, but, for particular applications, it has been shown to quickly produce optimal or near-optimal solutions (see references below).

**Note:** *The heuristic mode should only be used if the solver is first trained on a large and statistically representative set of training instances.*


Current Limitations
-------------------

* Only binary and continuous decision variables are currently supported.

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
