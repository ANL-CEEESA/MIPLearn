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

The aggressiveness of classification components, such as `PrimalSolutionComponent` and `LazyConstraintComponent`, can be adjusted through the `threshold` constructor argument. Internally, these components ask the machine learning models how confident are they on each prediction they make, then automatically discard all predictions that have low confidence. The `threshold` argument specifies how confident should the ML models be for a prediction to be considered trustworthy. Lowering a component's threshold increases its aggressiveness, while raising a component's threshold makes it more conservative.

For example, if the ML model predicts that a certain binary variable will assume value `1.0` in the optimal solution with 75% confidence, and if the `PrimalSolutionComponent` is configured to discard all predictions with less than 90% confidence, then this variable will not be included in the predicted MIP start.

MIPLearn currently provides two types of thresholds:

* `MinProbabilityThreshold(p: List[float])` A threshold which indicates that a prediction is trustworthy if its probability of being correct, as computed by the machine learning model, is above a fixed value.
* `MinPrecisionThreshold(p: List[float])` A dynamic threshold which automatically adjusts itself during training to ensure that the component achieves at least a given precision on the training data set. Note that increasing a component's precision may reduce its recall.

The example below shows how to build a `PrimalSolutionComponent` which fixes variables to zero with at least 80% precision, and to one with at least 95% precision. Other components are configured similarly.

```python
from miplearn import PrimalSolutionComponent, MinPrecisionThreshold

PrimalSolutionComponent(
    mode="heuristic",
    threshold=lambda: MinPrecisionThreshold([0.80, 0.95]),
)
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

By default, given a training set of instantes, MIPLearn trains a fixed set of ML classifiers and regressors, then selects the best one based on cross-validation performance. Alternatively, the user may specify which ML model a component should use through the `classifier` or `regressor` contructor parameters. Scikit-learn classifiers and regressors are currently supported. A future version of the package will add compatibility with Keras models.

The example below shows how to construct a `PrimalSolutionComponent` which internally uses scikit-learn's `KNeighborsClassifiers`. Any other scikit-learn classifier or pipeline can be used. The classifier needs to be provided as a lambda function because the component may need to create multiple copies of it. It needs to be wrapped in `ScikitLearnClassifier` to ensure that all the proper data transformations are applied.

```python
from miplearn import PrimalSolutionComponent, ScikitLearnClassifier
from sklearn.neighbors import KNeighborsClassifier

comp = PrimalSolutionComponent(
    classifier=lambda: ScikitLearnClassifier(
        KNeighborsClassifier(n_neighbors=5),
    ),
)
comp.fit(train_instances)
``` 
