# MIPLearn

**MIPLearn** is an extensible framework for solving discrete optimization problems using a combination of Mixed-Integer Linear Programming (MIP) and Machine Learning (ML). The framework uses ML methods to automatically identify patterns in previously solved instances of the problem, then uses these patterns to accelerate the performance of conventional state-of-the-art MIP solvers (such as CPLEX, Gurobi or XPRESS).

Unlike pure ML methods, MIPLearn is not only able to find high-quality solutions to discrete optimization problems, but it can also prove the optimality and feasibility of these solutions.
Unlike conventional MIP solvers, MIPLearn can take full advantage of very specific observations that happen to be true in a particular family of instances (such as the observation that a particular constraint is typically redundant, or that a particular variable typically assumes a certain value). 

For certain classes of problems, this approach has been shown to provide significant performance benefits (see [benchmarks](benchmark.md) and [references](about.md)).

## Features

* **MIPLearn proposes a flexible problem specification format,** which allows users to describe their particular optimization problems to a Learning-Enhanced MIP solver, both from the MIP perspective and from the ML perspective, without making any assumptions on the problem being modeled, the mathematical formulation of the problem, or ML encoding. While the format is very flexible, some constraints are enforced to ensure that it is usable by an actual solver.

* **MIPLearn provides a reference implementation of a *Learning-Enhanced Solver*,** which can use the above problem specification format to automatically predict, based on previously solved instances, a number of hints to accelerate MIP performance. Currently, the reference solver is able to predict: (i) partial solutions which are likely to work well as MIP starts; (ii) an initial set of lazy constraints to enforce; (iii) variable branching priorities to accelerate the exploration of the branch-and-bound tree; (iv) the optimal objective value based on the solution to the LP relaxation. The usage of the solver is very straightforward. The most suitable ML models are automatically selected, trained, cross-validated and applied to the problem with no user intervention.

* **MIPLearn provides a set of benchmark problems and random instance generators,** covering applications from different domains, which can be used to quickly evaluate new learning-enhanced MIP techniques in a measurable and reproducible way.

* **MIPLearn is customizable and extensible**. For MIP and ML researchers exploring new techniques to accelerate MIP performance based on historical data, each component of the reference solver can be individually replaced, extended or customized.

## Site contents

```{toctree}
---
maxdepth: 2
---
usage.md
benchmark.md
customization.md
about.md
```

## Source Code

* [https://github.com/ANL-CEEESA/MIPLearn](https://github.com/ANL-CEEESA/MIPLearn)

