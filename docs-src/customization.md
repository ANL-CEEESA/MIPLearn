# Customization


### Selecting the internal MIP solver

By default, `LearningSolver` uses [Gurobi](https://www.gurobi.com/) as its internal MIP solver. Alternative solvers can be specified through the `internal_solver_factory` constructor argument. This argument should provide a function (with no arguments) that constructs, configures and returns the desired solver. To select CPLEX, for example:
```python
from miplearn import LearningSolver
import pyomo.environ as pe

def cplex_factory():
    cplex = pe.SolverFactory("cplex")
    cplex.options["threads"] = 4
    return cplex

solver = LearningSolver(internal_solver_factory=cplex_factory)
```
