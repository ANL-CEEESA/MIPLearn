# MIPLearn

**MIPLearn** is an extensible framework for solving discrete optimization problems using a combination of Mixed-Integer Linear Programming (MIP) and Machine Learning (ML). The framework uses ML methods to automatically identify patterns in previously solved instances of the problem, then uses these patterns to accelerate the performance of conventional state-of-the-art MIP solvers (such as CPLEX, Gurobi or XPRESS).

Unlike pure ML methods, MIPLearn is not only able to find high-quality solutions to discrete optimization problems, but it can also prove the optimality and feasibility of these solutions.
Unlike conventional MIP solvers, MIPLearn can take full advantage of very specific observations that happen to be true in a particular family of instances (such as the observation that a particular constraint is typically redundant, or that a particular variable typically assumes a certain value). 

## Table of Contents

```{toctree}
---
maxdepth: 1
caption: Julia Tutorials
numbered: true
---
jump-tutorials/getting-started.ipynb
#jump-tutorials/lazy-constraints.ipynb
#jump-tutorials/user-cuts.ipynb
#jump-tutorials/customizing-ml.ipynb
```

```{toctree}
---
maxdepth: 1
caption: Benchmarks
numbered: true
---
benchmarks/preliminaries.ipynb
benchmarks/stab.ipynb
#benchmarks/uc.ipynb
#benchmarks/facility.ipynb
benchmarks/knapsack.ipynb
benchmarks/tsp.ipynb
```


```{toctree}
---
maxdepth: 1
caption: MIPLearn Internals
numbered: true
---
#internals/solver-interfaces.ipynb
#internals/data-collection.ipynb
#internals/abstract-component.ipynb
#internals/primal.ipynb
#internals/static-lazy.ipynb
#internals/dynamic-lazy.ipynb
```

## Source Code

* [https://github.com/ANL-CEEESA/MIPLearn](https://github.com/ANL-CEEESA/MIPLearn)

## Authors

* **Alinson S. Xavier,** Argonne National Laboratory <<axavier@anl.gov>>
* **Feng Qiu,** Argonne National Laboratory <<fqiu@anl.gov>>

## Acknowledgments

* Based upon work supported by U.S. Department of Energy **Advanced Grid Modeling Program** under Grant DE-OE0000875.
* Based upon work supported by **Laboratory Directed Research and Development** (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357

## References


If you use MIPLearn in your research, or the included problem generators, we kindly request that you cite the package as follows:
- **Alinson S. Xavier, Feng Qiu.** *MIPLearn: An Extensible Framework for Learning-Enhanced Optimization*. Zenodo (2020). DOI: [10.5281/zenodo.4287567](https://doi.org/10.5281/zenodo.4287567)

If you use MIPLearn in the field of power systems optimization, we kindly request that you cite the reference below, in which the main techniques implemented in MIPLearn were first developed:
- **Alinson S. Xavier, Feng Qiu, Shabbir Ahmed.** *Learning to Solve Large-Scale Unit Commitment Problems.* INFORMS Journal on Computing (2021). DOI: [10.1287/ijoc.2020.0976](https://doi.org/10.1287/ijoc.2020.0976)

## License

```text
MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
Copyright © 2020, UChicago Argonne, LLC. All Rights Reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
   conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials provided
   with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific prior written
   permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
