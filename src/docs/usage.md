# Usage


### Installation

MIPLearn is mainly written in Python, with some components written in Julia. For this
reason, both Python 3.6+ and Julia 1.3+ are required. A mixed-integer solver is also required, and
its Python bindings must be properly installed. Currently supported solvers include CPLEX and
Gurobi. To install MIPLearn, run the following commands: 

```bash
git clone https://github.com/ANL-CEEESA/MIPLearn.git
cd MIPLearn
make install
```

After installation, the package `miplearn` should become available to Python. It can be imported
as follows:

```python
import miplearn
```

!!! note
    To install MIPLearn in another Python environment, switch to that enviroment before running `make install`. To install the package in development mode, run `make develop` instead.

### Using `LearningSolver`

The main class provided by this package is `LearningSolver`, a learning-enhanced MIP solver which uses information from previously solved instances to accelerate the solution of new instances. The following example shows its basic usage:

```python
from miplearn import LearningSolver

# List of user-provided instances
training_instances = [...] 
test_instances = [...]

# Create solver
solver = LearningSolver()

# Solve all training instances
for instance in training_instances:
    solver.solve(instance)

# Learn from training instances
solver.fit(training_instances)

# Solve all test instances
for instance in test_instances:
    solver.solve(instance)
```

In this example, we have two lists of user-provided instances: `training_instances` and `test_instances`. We start by solving all training instances. Since there is no historical information available at this point, the instances will be processed from scratch, with no ML acceleration. After solving each instance, the solver stores within each `instance` object the optimal solution, the optimal objective value, and other information that can be used to accelerate future solves. After all training instances are solved, we call `solver.fit(training_instances)`. This instructs the solver to train all its internal machine-learning models based on the solutions of the (solved) trained instances. Subsequent calls to `solver.solve(instance)` will automatically use the trained Machine Learning models to accelerate the solution process.


### Describing problem instances

Instances to be solved by `LearningSolver` must derive from the abstract class `miplearn.Instance`. The following three abstract methods must be implemented:

* `instance.to_model()`, which returns a concrete Pyomo model corresponding to the instance;
* `instance.get_instance_features()`, which returns a 1-dimensional Numpy array of (numerical) features describing the entire instance;
* `instance.get_variable_features(var_name, index)`, which returns a 1-dimensional array of (numerical) features describing a particular decision variable.

The first method is used by `LearningSolver` to construct a concrete Pyomo model, which will be provided to the internal MIP solver. The second and third methods provide an encoding of the instance, which can be used by the ML models to make predictions. In the knapsack problem, for example, an implementation may decide to provide as instance features the average weights, average prices, number of items and the size of the knapsack. The weight and the price of each individual item could be provided as variable features. See `src/python/miplearn/problems/knapsack.py` for a concrete example.

An optional method which can be implemented is `instance.get_variable_category(var_name, index)`, which returns a category (a string, an integer or any hashable type) for each decision variable. If two variables have the same category, `LearningSolver` will use the same internal ML model to predict the values of both variables. By default, all variables belong to the `"default"` category, and therefore only one ML model is used for all variables. If the returned category is `None`, ML predictors will ignore the variable.

It is not necessary to have a one-to-one correspondence between features and problem instances. One important (and deliberate) limitation of MIPLearn, however, is that `get_instance_features()` must always return arrays of same length for all relevant instances of the problem. Similarly, `get_variable_features(var_name, index)` must also always return arrays of same length for all variables in each category. It is up to the user to decide how to encode variable-length characteristics of the problem into fixed-length vectors. In graph problems, for example, graph embeddings can be used to reduce the (variable-length) lists of nodes and edges into a fixed-length structure that still preserves some properties of the graph. Different instance encodings may have significant impact on performance.


### Obtaining heuristic solutions

By default, `LearningSolver` uses Machine Learning to accelerate the MIP solution process, while maintaining all optimality guarantees provided by the MIP solver. In the default mode of operation, for example, predicted optimal solutions are used only as MIP starts.

For more significant performance benefits, `LearningSolver` can also be configured to place additional trust in the Machine Learning predictors, by using the `mode="heuristic"` constructor argument. When operating in this mode, if a ML model is statistically shown (through *stratified k-fold cross validation*) to have exceptionally high accuracy, the solver may decide to restrict the search space based on its predictions. The parts of the solution which the ML models cannot predict accurately will still be explored using traditional (branch-and-bound) methods.  For particular applications, this mode has been shown to quickly produce optimal or near-optimal solutions (see [references](about.md#references) and [benchmark results](benchmark.md)).


!!! danger
    The `heuristic` mode provides no optimality guarantees, and therefore should only be used if the solver is first trained on a large and representative set of training instances. Training on a small or non-representative set of instances may produce low-quality solutions, or make the solver incorrectly classify new instances as infeasible.


### Saving and loading solver state

After solving a large number of training instances, it may be desirable to save the current state of `LearningSolver` to disk, so that the solver can still use the acquired knowledge after the application restarts. This can be accomplished by using the standard `pickle` module, as the following example illustrates:

```python
from miplearn import LearningSolver
import pickle

# Solve training instances
training_instances = [...]
solver = LearningSolver()
for instance in training_instances:
    solver.solve(instance)

# Train machine-learning models
solver.fit(training_instances)

# Save trained solver to disk
pickle.dump(solver, open("solver.pickle", "wb"))

# Application restarts...

# Load trained solver from disk
solver = pickle.load(open("solver.pickle", "rb"))

# Solve additional instances
test_instances = [...]
for instance in test_instances:
    solver.solve(instance)
```


### Solving training instances in parallel

In many situations, training and test instances can be solved in parallel to accelerate the training process. `LearningSolver` provides the method `parallel_solve(instances)` to easily achieve this:

```python
from miplearn import LearningSolver

training_instances = [...]
solver = LearningSolver()
solver.parallel_solve(training_instances, n_jobs=4)
solver.fit(training_instances)

# Test phase...
test_instances = [...]
solver.parallel_solve(test_instances)
```


### Current Limitations

* Only binary and continuous decision variables are currently supported.
