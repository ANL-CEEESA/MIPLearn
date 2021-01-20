# Customization

## Customizing solver parameters

### Selecting the internal MIP solver

By default, `LearningSolver` uses [Gurobi](https://www.gurobi.com/) as its internal MIP solver, and expects models to be provided using the Pyomo modeling language. Supported solvers and modeling languages include:

* `GurobiPyomoSolver`: Gurobi with Pyomo (default).
* `CplexPyomoSolver`: [IBM ILOG CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) with Pyomo.
* `XpressPyomoSolver`: [FICO XPRESS Solver](https://www.fico.com/en/products/fico-xpress-solver) with Pyomo.
* `GurobiSolver`: Gurobi without any modeling language.

To switch between solvers, provide the desired class using the `solver` argument:

```python
from miplearn import LearningSolver, CplexPyomoSolver
solver = LearningSolver(solver=CplexPyomoSolver)
```

To configure a particular solver, use the `params` constructor argument, as shown below.

```python
from miplearn import LearningSolver, GurobiPyomoSolver
solver = LearningSolver(
    solver=lambda: GurobiPyomoSolver(
        params={
            "TimeLimit": 900,
            "MIPGap": 1e-3,
            "NodeLimit": 1000,
        }
    ),
)
```


## Customizing solver components

`LearningSolver` is composed by a number of individual machine-learning components, each targeting a different part of the solution process. Each component can be individually enabled, disabled or customized. The following components are enabled by default:

* `LazyConstraintComponent`: Predicts which lazy constraint to initially enforce.
* `ObjectiveValueComponent`: Predicts the optimal value of the optimization problem, given the optimal solution to the LP relaxation.
* `PrimalSolutionComponent`: Predicts optimal values for binary decision variables. In heuristic mode, this component fixes the variables to their predicted values. In exact mode, the predicted values are provided to the solver as a (partial) MIP start.

The following components are also available, but not enabled by default:

* `BranchPriorityComponent`: Predicts good branch priorities for decision variables.

### Selecting components

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

### Adjusting component aggressiveness

The aggressiveness of classification components (such as `PrimalSolutionComponent` and `LazyConstraintComponent`) can
be adjusted through the `threshold` constructor argument. Internally, these components ask the ML models how confident
they are on each prediction (through the `predict_proba` method in the sklearn API), and only take into account
predictions which have probabilities above the threshold. Lowering a component's threshold increases its aggressiveness,
while raising a component's threshold makes it more conservative. 

MIPLearn also includes `MinPrecisionThreshold`, a dynamic threshold which adjusts itself automatically during training
to achieve a minimum desired true positive rate (also known as precision). The example below shows how to initialize
a `PrimalSolutionComponent` which achieves 95% precision, possibly at the cost of a lower recall. To make the component
more aggressive, this precision may be lowered.

```python
PrimalSolutionComponent(threshold=MinPrecisionThreshold(0.95))
```

### Evaluating component performance

MIPLearn allows solver components to be modified, trained and evaluated in isolation. In the following example, we build and
fit `PrimalSolutionComponent` outside the solver, then evaluate its performance.

```python
from miplearn import PrimalSolutionComponent

# User-provided set of previously-solved instances
train_instances = [...]

# Construct and fit component on a subset of training instances
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
```text
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

Regression components (such as `ObjectiveValueComponent`) can also be trained and evaluated similarly,
as the next example shows:

```python
from miplearn import ObjectiveValueComponent
comp = ObjectiveValueComponent()
comp.fit(train_instances[:100])
ev = comp.evaluate(train_instances[100:150])

import pandas as pd
pd.DataFrame(ev).mean(axis=1)
```
```text
Mean squared error       7001.977827
Explained variance          0.519790
Max error                 242.375804
Mean absolute error        65.843924
R2                          0.517612
Median absolute error      65.843924
dtype: float64
```

### Using customized ML classifiers and regressors

By default, given a training set of instantes, MIPLearn trains a fixed set of ML classifiers and regressors, then
selects the best one based on cross-validation performance. Alternatively, the user may specify which ML model a component
should use through the `classifier` or `regressor` contructor parameters. The provided classifiers and regressors must 
follow the sklearn API. In particular, classifiers must provide the methods `fit`, `predict_proba` and `predict`,
while regressors must provide the methods `fit` and `predict`

!!! danger
    MIPLearn must be able to generate a copy of any custom ML classifiers and regressors through 
    the standard  `copy.deepcopy` method. This currently makes it incompatible with Keras and TensorFlow
    predictors. This is a known limitation, which will be addressed in a future version.
    
The example below shows how to construct a `PrimalSolutionComponent` which internally uses
sklearn's `KNeighborsClassifiers`. Any other sklearn classifier or pipeline can be used. 

```python
from miplearn import PrimalSolutionComponent
from sklearn.neighbors import KNeighborsClassifier

comp = PrimalSolutionComponent(classifier=KNeighborsClassifier(n_neighbors=5))
comp.fit(train_instances)
``` 
  