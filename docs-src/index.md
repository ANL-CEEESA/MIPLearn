# MIPLearn

**MIPLearn** is an extensible framework for *Learning-Enhanced Mixed-Integer Optimization*, an approach targeted at discrete optimization problems that need to be repeatedly solved with only minor changes to input data. The package uses Machine Learning (ML) to automatically identify patterns in previously solved instances of the problem, or in the solution process itself, and produces hints that can guide a conventional MIP solver towards the optimal solution faster. For particular classes of problems, this approach has been shown to provide significant performance benefits (see [benchmark results](benchmark.md#benchmark-results) and [references](about.md#references) for more details).

### Features

* **MIPLearn proposes a flexible problem specification format,** which allows users to describe their particular optimization problems to a Learning-Enhanced MIP solver, both from the MIP perspective and from the ML perspective, without making any assumptions on the problem being modeled, the mathematical formulation of the problem, or ML encoding. While the format is very flexible, some constraints are enforced to ensure that it is usable by an actual solver.

* **MIPLearn provides a reference implementation of a *Learning-Enhanced Solver*,** which can use the above problem specification format to automatically predict, based on previously solved instances, a number of hints to accelerate MIP performance. Currently, the reference solver is able to predict: (i) partial solutions which are likely to work well as MIP starts; (ii) an initial set of lazy constraints to enforce; (iii) affine subspaces where the solution is likely to reside; (iv) variable branching priorities to accelerate the exploration of the branch-and-bound tree. The usage of the solver is very straightforward. The most suitable ML models are automatically selected, trained, cross-validated and applied to the problem with no user intervention.

* **MIPLearn provides a set of benchmark problems and random instance generators,** covering applications from different domains, which can be used to quickly evaluate new learning-enhanced MIP techniques in a measurable and reproducible way.

* **MIPLearn is customizable and extensible**. For MIP and ML researchers exploring new techniques to accelerate MIP performance based on historical data, each component of the reference solver can be individually replaced, extended or customized.

### Documentation

* [Installation and typical usage](usage.md)
* [Benchmark utilities](benchmark.md)
* [Benchmark problems, challenges and results](problems.md)
* [Customizing the solver](customization.md)
* [License, authors, references and acknowledgements](about.md)

### Souce Code

* [https://github.com/iSoron/miplearn](https://github.com/iSoron/miplearn)

