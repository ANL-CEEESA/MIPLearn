# Usage


### Installation

The package is currently available for Python and Pyomo. It can be installed as follows:

```bash
pip install git+ssh://git@github.com/ANL-CEEESA/MIPLearn.git
```

A Julia + JuMP version of the package is planned.


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
* `instance.get_variable_features(var_name, index)`, which returns a 1-dimensional array of (numerical) features describing a particular decision variable.


The first method is used by `LearningSolver` to construct a concrete Pyomo model, which will be provided to the internal MIP solver. The user should keep a reference to this Pyomo model, in order to retrieve, for example, the optimal variable values.

The second and third methods provide an encoding of the instance, which can be used by the ML models to make predictions. In the knapsack problem, for example, an implementation may decide to provide as instance features the average weights, average prices, number of items and the size of the knapsack. The weight and the price of each individual item could be provided as variable features. See `miplearn/problems/knapsack.py` for a concrete example.

An optional method which can be implemented is `instance.get_variable_category(var_name, index)`, which returns a category (a string, an integer or any hashable type) for each decision variable. If two variables have the same category, `LearningSolver` will use the same internal ML model to predict the values of both variables. By default, all variables belong to the `"default"` category, and therefore only one ML model is used for all variables. If the returned category is `None`, ML predictors will ignore the variable.

It is not necessary to have a one-to-one correspondence between features and problem instances. One important (and deliberate) limitation of MIPLearn, however, is that `get_instance_features()` must always return arrays of same length for all relevant instances of the problem. Similarly, `get_variable_features(var_name, index)` must also always return arrays of same length for all variables in each category. It is up to the user to decide how to encode variable-length characteristics of the problem into fixed-length vectors. In graph problems, for example, graph embeddings can be used to reduce the (variable-length) lists of nodes and edges into a fixed-length structure that still preserves some properties of the graph. Different instance encodings may have significant impact on performance.


### Obtaining heuristic solutions

By default, `LearningSolver` uses Machine Learning to accelerate the MIP solution process, while maintaining all optimality guarantees provided by the MIP solver. In the default mode of operation, for example, predicted optimal solutions are used only as MIP starts.

For more significant performance benefits, `LearningSolver` can also be configured to place additional trust in the Machine Learning predictors, by using the `mode="heuristic"` constructor argument. When operating in this mode, if a ML model is statistically shown (through *stratified k-fold cross validation*) to have exceptionally high accuracy, the solver may decide to restrict the search space based on its predictions. The parts of the solution which the ML models cannot predict accurately will still be explored using traditional (branch-and-bound) methods.  For particular applications, this mode has been shown to quickly produce optimal or near-optimal solutions (see [references](about.md#references) and [benchmark results](benchmark.md)).


!!! danger
    The `heuristic` mode provides no optimality guarantees, and therefore should only be used if the solver is first trained on a large and representative set of training instances. Training on a small or non-representative set of instances may produce low-quality solutions, or make the solver incorrectly classify new instances as infeasible.


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


### Current Limitations

* Only binary and continuous decision variables are currently supported.
* Solver callbacks (lazy constraints, cutting planes) are not currently supported.
* Only Gurobi and CPLEX are currently supported as internal MIP solvers.