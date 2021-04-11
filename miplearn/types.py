#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Optional, Dict, Callable, Any, Union, Tuple, TYPE_CHECKING, Hashable

from mypy_extensions import TypedDict

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from miplearn.solvers.learning import InternalSolver

BranchPriorities = Dict[str, Optional[float]]
Category = Hashable
IterationCallback = Callable[[], bool]
LazyCallback = Callable[[Any, Any], None]
SolverParams = Dict[str, Any]
UserCutCallback = Callable[["InternalSolver", Any], None]
VariableName = str
Solution = Dict[VariableName, Optional[float]]

MIPSolveStats = TypedDict(
    "MIPSolveStats",
    {
        "Lower bound": Optional[float],
        "MIP log": str,
        "Nodes": Optional[int],
        "Sense": str,
        "Upper bound": Optional[float],
        "Wallclock time": float,
        "Warm start value": Optional[float],
    },
)

LearningSolveStats = TypedDict(
    "LearningSolveStats",
    {
        "Gap": Optional[float],
        "Instance": Union[str, int],
        "LP log": str,
        "LP value": Optional[float],
        "Lower bound": Optional[float],
        "MIP log": str,
        "Mode": str,
        "Nodes": Optional[int],
        "Objective: Predicted lower bound": float,
        "Objective: Predicted upper bound": float,
        "Primal: Free": int,
        "Primal: One": int,
        "Primal: Zero": int,
        "Sense": str,
        "Solver": str,
        "Upper bound": Optional[float],
        "Wallclock time": float,
        "Warm start value": Optional[float],
        "LazyStatic: Removed": int,
        "LazyStatic: Kept": int,
        "LazyStatic: Restored": int,
        "LazyStatic: Iterations": int,
        "UserCuts: Added ahead-of-time": int,
        "UserCuts: Added in callback": int,
    },
    total=False,
)
