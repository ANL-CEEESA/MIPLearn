<h1 align="center">MIPLearn</h1>
<p align="center">
  <a href="https://github.com/ANL-CEEESA/MIPLearn/actions">
    <img src="https://github.com/ANL-CEEESA/MIPLearn/workflows/Test/badge.svg">
  </a>
  <a href="https://doi.org/10.5281/zenodo.4287567">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4287567.svg">
  </a>
  <a href="https://github.com/ANL-CEEESA/MIPLearn/releases/">
    <img src="https://img.shields.io/github/v/release/ANL-CEEESA/MIPLearn?include_prereleases&label=pre-release">
  </a>
  <a href="https://github.com/ANL-CEEESA/MIPLearn/discussions">
    <img src="https://img.shields.io/badge/GitHub-Discussions-%23fc4ebc" />
  </a>
</p>

**MIPLearn** is an extensible framework for solving discrete optimization problems using a combination of Mixed-Integer Linear Programming (MIP) and Machine Learning (ML). MIPLearn uses ML methods to automatically identify patterns in previously solved instances of the problem, then uses these patterns to accelerate the performance of conventional state-of-the-art MIP solvers such as CPLEX, Gurobi or XPRESS.

Unlike pure ML methods, MIPLearn is not only able to find high-quality solutions to discrete optimization problems, but it can also prove the optimality and feasibility of these solutions. Unlike conventional MIP solvers, MIPLearn can take full advantage of very specific observations that happen to be true in a particular family of instances (such as the observation that a particular constraint is typically redundant, or that a particular variable typically assumes a certain value). For certain classes of problems, this approach may provide significant performance benefits.

Documentation
-------------

- Tutorials:
    1. [Getting started (Pyomo)](https://anl-ceeesa.github.io/MIPLearn/0.4/tutorials/getting-started-pyomo/)
    2. [Getting started (Gurobipy)](https://anl-ceeesa.github.io/MIPLearn/0.4/tutorials/getting-started-gurobipy/)
    3. [Getting started (JuMP)](https://anl-ceeesa.github.io/MIPLearn/0.4/tutorials/getting-started-jump/)
    4. [User cuts and lazy constraints](https://anl-ceeesa.github.io/MIPLearn/0.4/tutorials/cuts-gurobipy/)
- User Guide
    1. [Benchmark problems](https://anl-ceeesa.github.io/MIPLearn/0.4/guide/problems/)
    2. [Training data collectors](https://anl-ceeesa.github.io/MIPLearn/0.4/guide/collectors/)
    3. [Feature extractors](https://anl-ceeesa.github.io/MIPLearn/0.4/guide/features/)
    4. [Primal components](https://anl-ceeesa.github.io/MIPLearn/0.4/guide/primal/)
    5. [Learning solver](https://anl-ceeesa.github.io/MIPLearn/0.4/guide/solvers/)
- Python API Reference
    1. [Benchmark problems](https://anl-ceeesa.github.io/MIPLearn/0.4/api/problems/)
    2. [Collectors & extractors](https://anl-ceeesa.github.io/MIPLearn/0.4/api/collectors/)
    3. [Components](https://anl-ceeesa.github.io/MIPLearn/0.4/api/components/)
    4. [Solvers](https://anl-ceeesa.github.io/MIPLearn/0.4/api/solvers/)
    5. [Helpers](https://anl-ceeesa.github.io/MIPLearn/0.4/api/helpers/)

Authors
-------

- **Alinson S. Xavier** (Argonne National Laboratory)
- **Feng Qiu** (Argonne National Laboratory)
- **Xiaoyi Gu** (Georgia Institute of Technology)
- **Berkay Becu** (Georgia Institute of Technology)
- **Santanu S. Dey**  (Georgia Institute of Technology)


Acknowledgments
---------------
* Based upon work supported by **Laboratory Directed Research and Development** (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy.
* Based upon work supported by the **U.S. Department of Energy Advanced Grid Modeling Program**.

Citing MIPLearn
---------------

If you use MIPLearn in your research (either the solver or the included problem generators), we kindly request that you cite the package as follows:

* **Alinson S. Xavier, Feng Qiu, Xiaoyi Gu, Berkay Becu, Santanu S. Dey.** *MIPLearn: An Extensible Framework for Learning-Enhanced Optimization (Version 0.4)*. Zenodo (2024). DOI: [10.5281/zenodo.4287567](https://doi.org/10.5281/zenodo.4287567)

If you use MIPLearn in the field of power systems optimization, we kindly request that you cite the reference below, in which the main techniques implemented in MIPLearn were first developed:

* **Alinson S. Xavier, Feng Qiu, Shabbir Ahmed.** *Learning to Solve Large-Scale Unit Commitment Problems.* INFORMS Journal on Computing (2020). DOI: [10.1287/ijoc.2020.0976](https://doi.org/10.1287/ijoc.2020.0976)

License
-------

Released under the modified BSD license. See `LICENSE` for more details.

