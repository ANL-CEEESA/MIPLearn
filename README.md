MIPLearn
========

**MIPLearn** is an extensible framework for *Learning-Enhanced Mixed-Integer Optimization*, an approach targeted at discrete optimization problems that need to be repeatedly solved with only minor changes to input data. The package uses Machine Learning (ML) to automatically identify patterns in previously solved instances of the problem, or in the solution process itself, and produces hints that can guide a conventional MIP solver towards the optimal solution faster. For particular classes of problems, this approach has been shown to provide significant performance benefits.

Features
--------
* **MIPLearn proposes a flexible problem specification format,** which allows users to describe their particular optimization problems to a Learning-Enhanced MIP solver, both from the MIP perspective and from the ML perspective, without making any assumptions on the problem being modeled, the mathematical formulation of the problem, or ML encoding. While the format is very flexible, some constraints are enforced to ensure that it is usable by an actual solver.

* **MIPLearn provides a reference implementation of a *Learning-Enhanced Solver*,** which can use the above problem specification format to automatically predict, based on previously solved instances, a number of hints to accelerate MIP performance. Currently, the reference solver is able to predict: (i) partial solutions which are likely to work well as MIP starts; (ii) an initial set of lazy constraints to enforce; (iii) affine subspaces where the solution is likely to reside; (iv) variable branching priorities to accelerate the exploration of the branch-and-bound tree. The usage of the solver is very straightforward. The most suitable ML models are automatically selected, trained, cross-validated and applied to the problem with no user intervention.

* **MIPLearn provides a set of benchmark problems and random instance generators,** covering applications from different domains, which can be used to quickly evaluate new learning-enhanced MIP techniques in a measurable and reproducible way.

* **MIPLearn is customizable and extensible**. For MIP and ML researchers exploring new techniques to accelerate MIP performance based on historical data, each component of the reference solver can be individually replaced, extended or customized.

Documentation
-------------

For installation instructions, basic usage and benchmarks results, see the official documentation at:

* [http://axavier.org/projects/miplearn](http://axavier.org/projects/miplearn)

License
-------

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

