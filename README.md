MIPLearn
========

**MIPLearn** is an optimization package designed to efficiently handle discrete optimization problems that need to be repeatedly solved with only relatively minor changes to input data.

The package uses Machine Learning techniques to extract information from previously solved instances and uses this information, when solving new instances, to construct a number of hints which can help a traditional Mixed-Integer Programming (MIP) solver, such as CPLEX or Gurobi, to more efficiently find the optimal solution and prove its optimality. For particular classes of problems, this approach has been shown to provide significant performance benefits (see references below).

<sub>This software is in alpha stage. Expect bugs and incomplete features. Suggestions and pull requests are very welcome.</sub>

Features
--------
* **MIPLearn provides a reference *Learning-Enhanced MIP Solver*,** which can automatically predict, based on previously solved instances (i) partial solutions which are likely to work well as MIP starts, (ii) an initial set of lazy constraints to enforce and (iii) affine subspaces where the solution is likely to reside. The reference solver can internally use CPLEX or Gurobi.

* **MIPLearn provides a flexible, problem-agnostic and formulation-agnostic framework** that allows users to model their own optimization problems. MIP formulations are provided to the learning-enhanced solver as [Pyomo](https://www.pyomo.org/) models, while features describing particular instances are provided as [NumPy](https://numpy.org/) arrays. The user can experiment with different MIP formulations and ML encodings.

* **MIPLearn provides a set of benchmark instances,** with problems from different domains, which can be used to quickly evaluate new learning-enhanced MIP techniques in a measurable and reproducible way.

Installation
------------

The package is currently only available for Python+Pyomo, although a Julia+JuMP version is planned. It can be installed using `pip` as follows:

```bash
pip install git+ssh://git@github.com/iSoron/miplearn.git
```

Usage
-----

### LearningSolver

The main class provided by this package is `LearningSolver`, a reference Learning-Enhanced MIP solver which automatically extracts information from previous runs to accelerate the solution of new instances. Assuming we already have a list of instances to solve, `LearningSolver` can be called as follows:

```python
from miplearn import LearningSolver

all_instances = ... # user-provided list of instances to solve
solver = LearningSolver()
for instance in all_instances:
    solver.solve(instance)
    solver.fit()
```

During the first call to `solver.solve(instance)`, the instance will be solved from scratch, since no historical information is available. The solver, however, will already start gathering information to accelerate future solves. After each solve, we call `solver.fit()` to train all the internal Machine Learning models based on the information gathered so far. As this operation can be expensive, it may also be performed after a larger batch of instances has been solved, instead of after every solve. After the first call to `solver.fit()`, subsequent calls to `solver.solve(instance)` will automatically use the trained Machine Learning models to accelerate the solution process.

By default, `LearningSolver` uses Cbc as its internal MIP solver. Alternative solvers can be specified through the `parent_solver`a argument, as follows. Persistent Pyomo solvers are supported.

```python
from miplearn import LearningSolver
import pyomo.environ as pe

solver = LearningSolver(parent_solver=pe.SolverFactory("gurobi_persistent"))
```

### Describing problem instances

Instances to be solved by `LearningSolver` must derive from the abstract class `miplearn.Instance`. The following three abstract methods must be implemented:

* `instance.to_model()`, which returns a concrete Pyomo model corresponding to the instance;
* `instance.get_instance_features()`, which returns a 1-dimensional Numpy array of (numerical) features describing the entire instance;
* `instance.get_variable_features(var, index)`, which returns a 1-dimensional array of (numerical) features describing a particular decision variable.


The first method is used by `LearningSolver` to construct a concrete Pyomo model, which will be provided to the internal MIP solver. The user should keep a reference to this Pyomo model, in order to retrieve, for example, the optimal variable values.

The second and third methods provide an encoding of the instance, which can be used by the ML models to make predictions. In the knapsack problem, for example, an implementation may decide to provide as instance features the average weights, average prices, number of items and the size of the knapsack. The weight and the price of each individual item could be provided as variable features. It is not necessary to have a one-to-one correspondence between features and problem instances.

An optional method which can be implemented is `instance.get_variable_category(var, index)`, which assigns a category for each decision variable. If two variables belong to the same category, `LearningSolver` will use the same internal ML model to predict the values of both variables. By default, all variables belong to the `"default"` category, and therefore only one ML model is used for all variables.

**Note:** Only binary and continuous decision variables are currently supported.

**Note:** One important (and deliberate) limitation of `MIPLearn` is that `get_instance_features()` must always return arrays of same length for all relevant instances of the problem. Similarly, `get_variable_features(var, index)` must also always return arrays of same length for all variables in each category. It is up to the user to decide how to encode variable-length characteristics of the problem into fixed-length vectors. In graph problems, for example, graph embeddings can be used to reduce the (variable-length) lists of nodes and edges into a fixed-length structure that still preserves some properties of the graph. Different instance encodings may have significant impact on performance.


References
----------

* **Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems.** *Alinson S. Xavier, Feng Qiu, Shabbir Ahmed*. INFORMS Journal on Computing (to appear). https://arxiv.org/abs/1902.01697

Authors
-------
* **Alinson S. Xavier,** Argonne National Laboratory <<axavier@anl.gov>>

License
-------

    MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
    Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
