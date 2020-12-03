![Build status](https://img.shields.io/github/workflow/status/ANL-CEEESA/MIPLearn/Test)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4287567.svg)](https://doi.org/10.5281/zenodo.4287567)
<a href="https://github.com/ANL-CEEESA/MIPLearn/releases/latest">
<img src="https://img.shields.io/github/v/release/ANL-CEEESA/MIPLearn">
</a>

MIPLearn
========

**MIPLearn** is an extensible framework for **Learning-Enhanced Mixed-Integer Optimization**, an approach targeted at discrete optimization problems that need to be repeatedly solved with only minor changes to input data.

The package uses Machine Learning (ML) to automatically identify patterns in previously solved instances of the problem, or in the solution process itself, and produces hints that can guide a conventional MIP solver towards the optimal solution faster. For particular classes of problems, this approach has been shown to provide significant performance benefits (see [benchmarks](https://anl-ceeesa.github.io/MIPLearn/0.1/problems/) and [references](https://anl-ceeesa.github.io/MIPLearn/0.1/about/)).

Features
--------
* **MIPLearn proposes a flexible problem specification format,** which allows users to describe their particular optimization problems to a Learning-Enhanced MIP solver, both from the MIP perspective and from the ML perspective, without making any assumptions on the problem being modeled, the mathematical formulation of the problem, or ML encoding.

* **MIPLearn provides a reference implementation of a *Learning-Enhanced Solver*,** which can use the above problem specification format to automatically predict, based on previously solved instances, a number of hints to accelerate MIP performance. 

* **MIPLearn provides a set of benchmark problems and random instance generators,** covering applications from different domains, which can be used to quickly evaluate new learning-enhanced MIP techniques in a measurable and reproducible way.

* **MIPLearn is customizable and extensible**. For MIP and ML researchers exploring new techniques to accelerate MIP performance based on historical data, each component of the reference solver can be individually replaced, extended or customized.

Documentation
-------------

For installation instructions, basic usage and benchmarks results, see the [official documentation](https://anl-ceeesa.github.io/MIPLearn/).

Acknowledgments
---------------
* Based upon work supported by **Laboratory Directed Research and Development** (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357, and the **U.S. Department of Energy Advanced Grid Modeling Program** under Grant DE-OE0000875.

Citing MIPLearn
---------------

If you use MIPLearn in your research (either the solver or the included problem generators), we kindly request that you cite the package as follows:

* **Alinson S. Xavier, Feng Qiu.** *MIPLearn: An Extensible Framework for Learning-Enhanced Optimization*. Zenodo (2020). DOI: [10.5281/zenodo.4287567](https://doi.org/10.5281/zenodo.4287567)

If you use MIPLearn in the field of power systems optimization, we kindly request that you cite the reference below, in which the main techniques implemented in MIPLearn were first developed:

* **Alinson S. Xavier, Feng Qiu, Shabbir Ahmed.** *Learning to Solve Large-Scale Unit Commitment Problems.* INFORMS Journal on Computing (2020). DOI: [10.1287/ijoc.2020.0976](https://doi.org/10.1287/ijoc.2020.0976)

License
-------

Released under the modified BSD license. See `LICENSE` for more details.

