# Customization

## Selecting the internal MIP solver

By default, `LearningSolver` uses [Gurobi](https://www.gurobi.com/) as its internal MIP solver. Another supported solver is [IBM ILOG CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio). To switch between solvers, use the `solver` constructor argument, as shown below. It is also possible to specify a time limit (in seconds) and a relative MIP gap tolerance.

```python
from miplearn import LearningSolver
solver = LearningSolver(solver="cplex",
                        time_limit=300,
                        gap_tolerance=1e-3)
```

## Selecting solver components

`LearningSolver` is composed by a number of individual machine-learning components, each targeting a different part of the solution process. Each component can be individually enabled, disabled or customized. The following components are enabled by default:

* `LazyConstraintComponent`: Predicts which lazy constraint to initially enforce.
* `ObjectiveValueComponent`: Predicts the optimal value of the optimization problem, given the optimal solution to the LP relaxation.
* `PrimalSolutionComponent`: Predicts optimal values for binary decision variables. In heuristic mode, this component fixes the variables to their predicted values. In exact mode, the predicted values are provided to the solver as a (partial) MIP start.

The following components are also available, but not enabled by default:

* `BranchPriorityComponent`: Predicts good branch priorities for decision variables.

To create a `LearningSolver` with a specific set of components, the `components` constructor argument may be used, as the next example shows:

```python
# Create a solver without any components
solver1 = LearningSolver(components=[])

# Create a solver with only two components
solver2 = LearningSolver(components=[
    LazyConstraintComponent(...),
    PrimalSolutionComponent(...),
])
```

It is also possible to add components to an existing solver using the `solver.add` method, as shown below. If the solver already holds another component of that type, the new component will replace the previous one.
```python
# Create solver with default components
solver = LearningSolver()

# Replace the default LazyConstraintComponent by one with custom parameters 
solver.add(LazyConstraintComponent(...))
```
