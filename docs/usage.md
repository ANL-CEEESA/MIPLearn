# Usage

## 1. Installation

In these docs, we describe the Python/Pyomo version of the package, although a [Julia/JuMP version](https://github.com/ANL-CEEESA/MIPLearn.jl) is also available. A mixed-integer solver is also required and its Python bindings must be properly installed. Supported solvers are currently CPLEX, Gurobi and XPRESS.

To install MIPLearn, run: 

```bash
pip3 install --upgrade miplearn==0.2.*
```

After installation, the package `miplearn` should become available to Python. It can be imported
as follows:

```python
import miplearn
```

## 2. Using `LearningSolver`

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


## 3. Describing problem instances

Instances to be solved by `LearningSolver` must derive from the abstract class `miplearn.Instance`. The following three abstract methods must be implemented:

* `instance.to_model()`, which returns a concrete Pyomo model corresponding to the instance;
* `instance.get_instance_features()`, which returns a 1-dimensional Numpy array of (numerical) features describing the entire instance;
* `instance.get_variable_features(var_name, index)`, which returns a 1-dimensional array of (numerical) features describing a particular decision variable.

The first method is used by `LearningSolver` to construct a concrete Pyomo model, which will be provided to the internal MIP solver. The second and third methods provide an encoding of the instance, which can be used by the ML models to make predictions. In the knapsack problem, for example, an implementation may decide to provide as instance features the average weights, average prices, number of items and the size of the knapsack. The weight and the price of each individual item could be provided as variable features. See `src/python/miplearn/problems/knapsack.py` for a concrete example.

An optional method which can be implemented is `instance.get_variable_category(var_name, index)`, which returns a category (a string, an integer or any hashable type) for each decision variable. If two variables have the same category, `LearningSolver` will use the same internal ML model to predict the values of both variables. By default, all variables belong to the `"default"` category, and therefore only one ML model is used for all variables. If the returned category is `None`, ML predictors will ignore the variable.

It is not necessary to have a one-to-one correspondence between features and problem instances. One important (and deliberate) limitation of MIPLearn, however, is that `get_instance_features()` must always return arrays of same length for all relevant instances of the problem. Similarly, `get_variable_features(var_name, index)` must also always return arrays of same length for all variables in each category. It is up to the user to decide how to encode variable-length characteristics of the problem into fixed-length vectors. In graph problems, for example, graph embeddings can be used to reduce the (variable-length) lists of nodes and edges into a fixed-length structure that still preserves some properties of the graph. Different instance encodings may have significant impact on performance.


## 4. Describing lazy constraints

For many MIP formulations, it is not desirable to add all constraints up-front, either because the total number of constraints is very large, or because some of the constraints, even in relatively small numbers, can still cause significant performance impact when added to the formulation. In these situations, it may be desirable to generate and add constraints incrementaly, during the solution process itself. Conventional MIP solvers typically start by solving the problem without any lazy constraints. Whenever a candidate solution is found, the solver finds all violated lazy constraints and adds them to the formulation. MIPLearn significantly accelerates this process by using ML to predict which lazy constraints should be enforced from the very beginning of the optimization process, even before a candidate solution is available.

MIPLearn supports two types of lazy constraints: through constraint annotations and through callbacks.

### 4.1 Adding lazy constraints through annotations

The easiest way to create lazy constraints in MIPLearn is to add them to the model (just like any regular constraints), then annotate them as lazy, as described below. Just before the optimization starts, MIPLearn removes all lazy constraints from the model and places them in a lazy constraint pool. If any trained ML models are available, MIPLearn queries these models to decide which of these constraints should be moved back into the formulation. After this step, the optimization starts, and lazy constraints from the pool are added to the model in the conventional fashion.

To tag a constraint as lazy, the following methods must be implemented:

* `instance.has_static_lazy_constraints()`, which returns `True` if the model has any annotated lazy constraints. By default, this method returns `False`.
* `instance.is_constraint_lazy(cid)`, which returns `True` if the constraint with name `cid` should be treated as a lazy constraint, and `False` otherwise.
* `instance.get_constraint_features(cid)`, which returns a 1-dimensional Numpy array of (numerical) features describing the constraint.

For instances such that `has_lazy_constraints` returns `True`, MIPLearn calls `is_constraint_lazy` for each constraint in the formulation, providing the name of the constraint. For constraints such that `is_constraint_lazy` returns `True`, MIPLearn additionally calls `get_constraint_features` to gather a ML representation of each constraint. These features are used to predict which lazy constraints should be initially enforced.

An additional method that can be implemented is `get_lazy_constraint_category(cid)`, which returns a category (a string or any other hashable type) for each lazy constraint. Similarly to decision variable categories, if two lazy constraints have the same category, then MIPLearn will use the same internal ML model to decide whether to initially enforce them. By default, all lazy constraints belong to the `"default"` category, and therefore a single ML model is used.

!!! warning
    If two lazy constraints belong to the same category, their feature vectors should have the same length.

### 4.2 Adding lazy constraints through callbacks

Although convenient, the method described in the previous subsection still requires the generation of all lazy constraints ahead of time, which can be prohibitively expensive. An alternative method is through a lazy constraint callbacks, described below. During the solution process, MIPLearn will repeatedly call a user-provided function to identify any violated lazy constraints. If violated constraints are identified, MIPLearn will additionally call another user-provided function to generate the constraint and add it to the formulation.

To describe lazy constraints through user callbacks, the following methods need to be implemented:

* `instance.has_dynamic_lazy_constraints()`, which returns `True` if the model has any lazy constraints generated by user callbacks. By default, this method returns `False`.
* `instance.find_violated_lazy_constraints(model)`, which returns a list of identifiers corresponding to the lazy constraints found to be violated by the current solution. These identifiers should be strings, tuples or any other hashable type.
* `instance.build_violated_lazy_constraints(model, cid)`, which returns either a list of Pyomo constraints, or a single Pyomo constraint, corresponding to the given lazy constraint identifier.
* `instance.get_constraint_features(cid)`, which returns a 1-dimensional Numpy array of (numerical) features describing the constraint. If this constraint is not valid, returns `None`.
* `instance.get_lazy_constraint_category(cid)`, which returns a category (a string or any other hashable type) for each lazy constraint, indicating which ML model to use. By default, returns `"default"`.


Assuming that trained ML models are available, immediately after calling `solver.solve`, MIPLearn will call `get_constraint_features` for each lazy constraint identifier found in the training set. For constraints such that `get_constraint_features` returns a vector (instead of `None`), MIPLearn will call `get_constraint_category` to decide which trained ML model to use. It will then query the ML model to decide whether the constraint should be initially enforced. Assuming that the ML predicts this constraint will be necessary, MIPLearn calls `build_violated_constraints` then adds the returned list of Pyomo constraints to the model. The optimization then starts. When no trained ML models are available, this entire initial process is skipped, and MIPLearn behaves like a conventional solver.

After the optimization process starts, MIPLearn will periodically call `find_violated_lazy_constraints` to verify if the current solution violates any lazy constraints. If any violated lazy constraints are found, MIPLearn will call the method `build_violated_lazy_constraints` and add the returned constraints to the formulation.

!!! note
    When implementing `find_violated_lazy_constraints(self, model)`, the current solution may be accessed through `self.solution[var_name][index]`.


## 5. Obtaining heuristic solutions

By default, `LearningSolver` uses Machine Learning to accelerate the MIP solution process, while maintaining all optimality guarantees provided by the MIP solver. In the default mode of operation, for example, predicted optimal solutions are used only as MIP starts.

For more significant performance benefits, `LearningSolver` can also be configured to place additional trust in the Machine Learning predictors, by using the `mode="heuristic"` constructor argument. When operating in this mode, if a ML model is statistically shown (through *stratified k-fold cross validation*) to have exceptionally high accuracy, the solver may decide to restrict the search space based on its predictions. The parts of the solution which the ML models cannot predict accurately will still be explored using traditional (branch-and-bound) methods.  For particular applications, this mode has been shown to quickly produce optimal or near-optimal solutions (see [references](about.md#references) and [benchmark results](benchmark.md)).


!!! danger
    The `heuristic` mode provides no optimality guarantees, and therefore should only be used if the solver is first trained on a large and representative set of training instances. Training on a small or non-representative set of instances may produce low-quality solutions, or make the solver incorrectly classify new instances as infeasible.

## 6. Scaling Up

### 6.1 Saving and loading solver state

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
with open("solver.pickle", "wb") as file:
    pickle.dump(solver, file)

# Application restarts...

# Load trained solver from disk
with open("solver.pickle", "rb") as file:
    solver = pickle.load(file)

# Solve additional instances
test_instances = [...]
for instance in test_instances:
    solver.solve(instance)
```


### 6.2 Solving instances in parallel

In many situations, instances can be solved in parallel to accelerate the training process. `LearningSolver` provides the method `parallel_solve(instances)` to easily achieve this:

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


### 6.3 Solving instances from the disk

In all examples above, we have assumed that instances are available as Python objects, stored in memory. When problem instances are very large, or when there is a large number of problem instances, this approach may require an excessive amount of memory. To reduce memory requirements, MIPLearn can also operate on instances that are stored on disk. More precisely, the methods `fit`, `solve` and `parallel_solve` in `LearningSolver` can operate on filenames (or lists of filenames) instead of instance objects, as the next example illustrates.
Instance files must be pickled instance objects. The method `solve` loads at most one instance to memory at a time, while `parallel_solve` loads at most `n_jobs` instances.


```python
import pickle
from miplearn import LearningSolver

# Construct and pickle 600 problem instances
for i in range(600):
    instance = MyProblemInstance([...])
    with open("instance_%03d.pkl" % i, "w") as file:
        pickle.dump(instance, obj)
        
# Split instances into training and test
test_instances  = ["instance_%03d.pkl" % i for i in range(500)]
train_instances = ["instance_%03d.pkl" % i for i in range(500, 600)]

# Create solver
solver = LearningSolver([...])

# Solve training instances 
solver.parallel_solve(train_instances, n_jobs=4)

# Train ML models
solver.fit(train_instances)

# Solve test instances 
solver.parallel_solve(test_instances, n_jobs=4)
```


By default, `solve` and `parallel_solve` modify files in place. That is, after the instances are loaded from disk and solved, MIPLearn writes them back to the disk, overwriting the original files. To write to an alternative file instead, use the arguments `output_filename` (in `solve`) and `output_filenames` (in `parallel_solve`). To discard the modifications instead, use `discard_outputs=True`. This can be useful, for example, during benchmarks.

```python
# Solve a single instance file and write the output to another file
solver.solve("knapsack_1.orig.pkl", output_filename="knapsack_1.solved.pkl")

# Solve a list of instance files
instances = ["knapsack_%03d.orig.pkl" % i for i in range(100)]
output = ["knapsack_%03d.solved.pkl" % i for i in range(100)]
solver.parallel_solve(instances, output_filenames=output)

# Solve instances and discard solutions and training data
solver.parallel_solve(instances, discard_outputs=True)
```

## 7. Running benchmarks

MIPLearn provides the utility class `BenchmarkRunner`, which simplifies the task of comparing the performance of different solvers. The snippet below shows its basic usage:

```python
from miplearn import BenchmarkRunner, LearningSolver

# Create train and test instances
train_instances = [...]
test_instances  = [...]

# Training phase...
training_solver = LearningSolver(...)
training_solver.parallel_solve(train_instances, n_jobs=10)

# Test phase...
benchmark = BenchmarkRunner({
    "Baseline": LearningSolver(...),
    "Strategy A": LearningSolver(...),
    "Strategy B": LearningSolver(...),
    "Strategy C": LearningSolver(...),
})
benchmark.fit(train_instances)
benchmark.parallel_solve(test_instances, n_jobs=5)
benchmark.write_csv("results.csv")
```

The method `fit` trains the ML models for each individual solver. The method `parallel_solve` solves the test instances in parallel, and collects solver statistics such as running time and optimal value. Finally, `write_csv` produces a table of results. The columns in the CSV file depend on the components added to the solver.

## 8. Current Limitations

* Only binary and continuous decision variables are currently supported. General integer variables are not currently supported by some solver components.
