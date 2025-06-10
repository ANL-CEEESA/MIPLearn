MIPLearn
========
**MIPLearn** is an extensible framework for solving discrete optimization problems using a combination of Mixed-Integer Linear Programming (MIP) and Machine Learning (ML). MIPLearn uses ML methods to automatically identify patterns in previously solved instances of the problem, then uses these patterns to accelerate the performance of conventional state-of-the-art MIP solvers such as CPLEX, Gurobi or XPRESS.

Unlike pure ML methods, MIPLearn is not only able to find high-quality solutions to discrete optimization problems, but it can also prove the optimality and feasibility of these solutions. Unlike conventional MIP solvers, MIPLearn can take full advantage of very specific observations that happen to be true in a particular family of instances (such as the observation that a particular constraint is typically redundant, or that a particular variable typically assumes a certain value). For certain classes of problems, this approach may provide significant performance benefits.


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :numbered: 2

   tutorials/getting-started-pyomo
   tutorials/getting-started-gurobipy
   tutorials/getting-started-jump
   tutorials/cuts-gurobipy

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :numbered: 2

   guide/problems
   guide/collectors
   guide/features
   guide/primal
   guide/solvers

.. toctree::
   :maxdepth: 1
   :caption: Python API Reference
   :numbered: 2

   api/problems
   api/collectors
   api/components
   api/solvers
   api/helpers


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

* **Alinson S. Xavier, Feng Qiu, Xiaoyi Gu, Berkay Becu, Santanu S. Dey.** *MIPLearn: An Extensible Framework for Learning-Enhanced Optimization (Version 0.4)*. Zenodo (2024). DOI: https://doi.org/10.5281/zenodo.4287567

If you use MIPLearn in the field of power systems optimization, we kindly request that you cite the reference below, in which the main techniques implemented in MIPLearn were first developed:

* **Alinson S. Xavier, Feng Qiu, Shabbir Ahmed.** *Learning to Solve Large-Scale Unit Commitment Problems.* INFORMS Journal on Computing (2020). DOI: https://doi.org/10.1287/ijoc.2020.0976
