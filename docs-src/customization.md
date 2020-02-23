# Customization


### Selecting the internal MIP solver

By default, `LearningSolver` uses [Gurobi](https://www.gurobi.com/) as its internal MIP solver. Another supported solver is [IBM ILOG CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio). To switch between solvers, use the `solver` constructor argument, as shown below. It is also possible to specify a time limit (in seconds) and a relative MIP gap tolerance.

```python
from miplearn import LearningSolver
solver = LearningSolver(solver="cplex",
                        time_limit=300,
                        gap_tolerance=1e-3)
```
