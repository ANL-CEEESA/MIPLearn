MIPLearn
========

**MIPLearn** is a software package for *Learning-Enhanced Mixed-Integer Linear Optimization*, aimed at efficiently handling discrete optimization problems that need to be solved repeatedly with only minor changes to input data. The package uses Machine Learning techniques to extract information from previously solved instances, and uses this information to improve the performance of traditional MIP solvers, such as CPLEX or Gurobi, on similar problem instances. For particular classes of problems, [this approach has been shown to provide significant performance benefits](https://arxiv.org/abs/1902.01697).

Features
--------
* **MIPLearn provides a reference implementation of a Learning-Enhanced MIP Solver,** which can automatically predict, based on previously solved instances: partial solutions which are likely to work well as MIP starts, an initial set of lazy constraints to enforce and affine subspaces where the solution is likely to reside. The reference solver can internally use CPLEX or Gurobi.

* **MIPLearn provides a flexible, problem-agnostic and formulation-agnostic framework,** that allows users to model their own optimization problems. MIP formulations are provided to the learning-enhanced solver as [Pyomo](https://www.pyomo.org/) models, while features describing particular instances are provided as [NumPy](https://numpy.org/) arrays. The user can experiment with different MIP formulations and ML encodings.

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

The main class provided by this package is `LearningSolver`, a reference Learning-Enhanced MIP solver which automatically extracts information from previous runs to accelerate the solution of new (yet unseen) instances. Assuming we already have a list of instances to solve, `LearningSolver` can be called as follows:

```python
from miplearn import LearningSolver

all_instances = ... # user-provided list of instances to solve
solver = LearningSolver()
for instance in all_instances:
    solver.solve(instance)
    solver.fit()
```

During the first call to `solver.solve(instance)`, the problem instance will be solved from scratch, since there is no historical data available. The solver, however, will already start gathering information to accelerate future solves. After each solve, we call `solver.fit()` to train all internal Machine Learning models based on the information gathered so far. As this operation can be expensive, it may also be performed after a larger batch of instances has been solved, instead of after every solve. After the first call to `solver.fit()`, subsequent calls to `solver.solve(instance)` will automatically use the trained Machine Learning models to accelerate the solution process.

### Describing problem instances

Instances provided to `LearningSolver` must implement the abstract class `miplearn.Instance`. The following three abstract methods must be implemented:

* `instance.to_model()`, which returns a concrete Pyomo model corresponding to this instance;
* `instance.get_instance_features()`, which returns a 1-dimensional Numpy array of (numerical) features describing the entire instance;
* `instance.get_variable_features(var, index)`, which returns a 1-dimensional array of (numerical) features describing a particular decision variable.


The first method is used by `LearningSolver` to construct a concrete Pyomo model, which will be provided to the underlying MIP solver. The user should keep a reference to this Pyomo model, in order to retrieve, for example, the optimal variable values. The second and third methods provide an encoding of the instance, which can be used by the ML models to make predictions. In the knapsack problem, for example, an implementation may decide to provide as instance features the average weights, average prices, number of items and the size of the knapsack. The weight and the price of each individual item could be provided as variable features. It is not necessary to have a one-to-one correspondence between features and problem instances.

An optional method which can be implemented is `instance.get_variable_category(var, index)`, which assigns a category for each decision variable. If two variables belong to the same category, `LearningSolver` will use the same internal ML model to predict the values of both variables. By default, all variables belong to the `"default"` category, and therefore only one ML model is used for all variables.

**Note:** One important (and deliberate) limitation of `MIPLearn` is that `get_instance_features()` must always return arrays of same length for all relevant instances of the problem. Similarly, `get_variable_features(var, index)` must also always return arrays of same length for all variables in each category. It is up to the user to decide how to encode variable-length characteristics of the problem into fixed-length vectors. In graph problems, for example, graph embeddings can be used to reduce the (variable-length) lists of nodes and edges into a fixed-length structure that still preserves some properties of the graph. Different instance encodings may have significant impact on performance.


References
----------

* **Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems.** *Alinson Santos Xavier, Feng Qiu, Shabbir Ahmed*. INFORMS Journal on Computing (to appear). https://arxiv.org/abs/1902.01697

Authors
-------
* **Alinson S. Xavier,** Argonne National Laboratory <<axavier@anl.gov>>

License
-------

    MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
    Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
