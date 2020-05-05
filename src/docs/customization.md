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

## Adjusting component aggresiveness

The aggressiveness of classification components (such as `PrimalSolutionComponent` and `LazyConstraintComponent`) can
be adjusted through the `threshold` constructor argument. Internally, these components ask the ML models how confident
they are on each prediction (through the `predict_proba` method in the sklearn API), and only take into account
predictions which have probabilities above the threshold. Lowering a component's threshold increases its aggresiveness,
while raising a component's threshold makes it more conservative. 

MIPLearn also includes `MinPrecisionThreshold`, a dynamic threshold which adjusts itself automatically during training
to achieve a minimum desired true positive rate (also known as precision). The example below shows how to initialize
a `PrimalSolutionComponent` which achieves 95% precision, possibly at the cost of a lower recall. To make the component
more aggressive, this precision may be lowered.

```python
PrimalSolutionComponent(threshold=MinPrecisionThreshold(0.95))
```

### Evaluating component performance

MIPLearn allows solver components to be modified and evaluated in isolation. In the following example, we build and
fit `PrimalSolutionComponent` outside a solver, then evaluate its performance.

```python
from miplearn import PrimalSolutionComponent

# User-provided set os solved training instances
train_instances = [...]

# Construct and fit component on a subset of the training set
comp = PrimalSolutionComponent()
comp.fit(train_instances[:100])

# Evaluate performance on an additional set of training instances
ev = comp.evaluate(train_instances[100:150])
``` 

The method `evaluate` returns a dictionary with performance evaluation statistics for each training instance provided,
and for each type of prediction the component makes. To obtain a summary across all instances, pandas may be used, as below:

```python
import pandas as pd
pd.DataFrame(ev["Fix one"]).mean(axis=1)
```
```
Predicted positive          3.120000
Predicted negative        196.880000
Condition positive         62.500000
Condition negative        137.500000
True positive               3.060000
True negative             137.440000
False positive              0.060000
False negative             59.440000
Accuracy                    0.702500
F1 score                    0.093050
Recall                      0.048921
Precision                   0.981667
Predicted positive (%)      1.560000
Predicted negative (%)     98.440000
Condition positive (%)     31.250000
Condition negative (%)     68.750000
True positive (%)           1.530000
True negative (%)          68.720000
False positive (%)          0.030000
False negative (%)         29.720000
dtype: float64
```

Regression components (such as `ObjectiveValueComponent`) can also be used similarly, as shown in the next example:

```python
from miplearn import ObjectiveValueComponent
comp = ObjectiveValueComponent()
comp.fit(train_instances[:100])
ev = comp.evaluate(train_instances[100:150])

import pandas as pd
pd.DataFrame(ev).mean(axis=1)
```
```
Mean squared error       7001.977827
Explained variance          0.519790
Max error                 242.375804
Mean absolute error        65.843924
R2                          0.517612
Median absolute error      65.843924
dtype: float64
```