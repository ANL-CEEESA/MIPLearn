# MIPLearn: Changelog

## [Unreleased]

### Added

- **Added two new machine learning components:**
  - Added `StaticLazyConstraintComponent`, which allows the user to mark some constraints in the formulation as lazy, instead of constructing them in a callback. ML predicts which static lazy constraints should be kept in the formulation, and which should be removed. 
  - Added `UserCutComponents`, which predicts which user cuts should be generated and added to the formulation as constraints ahead-of-time, before solving the MIP.
- **Added support to additional MILP solvers:**
  - Added support for CPLEX and XPRESS, through the Pyomo modeling language, in addition to (existing) Gurobi. The solver classes are named `CplexPyomoSolver`, `XpressPyomoSolver` and `GurobiPyomoSolver`.
  - Added support for Gurobi without any modeling language. The solver class is named `GurobiSolver`. In this case, `instance.to_model` should return ` gp.Model` object.
  - Added support to direct MPS files, produced externally, through the `GurobiSolver` class mentioned above.
- **Added dynamic thresholds:** 
  - In previous versions of the package, it was necessary to manually adjust component aggressiveness to reach a desired precision/recall. This can now be done automatically with `MinProbabilityThreshold`, `MinPrecisionThreshold` and `MinRecallThreshold`.
- **Reduced memory requirements:**
  - Previous versions of the package required all training instances to be kept in memory at all times, which was prohibitive for large-scale problems. It is now possible to store instances in file until they are needed, using `PickledGzInstance`.
- **Refactoring:**
  - Added static types to all classes (with mypy).

### Changed

- Variables are now referenced by their names, instead of tuples `(var_name, index)`. This change was required to improve the compatibility with modeling languages other than Pyomo, which do not follow this convention. The functions `get_variable_category` and `get_variable_features` now have the following signature:
  ````python
   def get_variable_features(self, var_name: str) -> List[float]:
      pass
  
   def get_variable_category(self, var_name: str) -> Optional[Hashable]:
      pass  
   ````
- Features are now represented as a list of floating point numbers, as indicated in the snippet above. This change was required for performance reasons. Returning numpy arrays is no longer supported, and raises an error.

- Internal solvers must now be specified as objects, instead of strings. For example,
  ```python
  solver = LearningSolver(
      solver=GurobiPyomoSolver(
          params={
              "TimeLimit": 300,
              "Threads": 4,
          }      
      )
  )
  ```
- `LazyConstraintComponent` has been renamed to `DynamicLazyConstraintsComponent`.

### Removed

- Temporarily removed the experimental `BranchPriorityComponent`. This component will be re-added in the Julia version of the package.
- Removed `solver.add` method, previously used to add components to an existing solver. Use the constructor `LearningSolver(components=[...])` instead.

## [0.1.0] - 2020-11-23

- Initial public release
