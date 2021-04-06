#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Optional, Dict, Callable, Any, Union, Tuple, TYPE_CHECKING

from mypy_extensions import TypedDict

if TYPE_CHECKING:
    from miplearn.solvers.learning import InternalSolver

VarIndex = Union[str, int, Tuple[Union[str, int]]]

Solution = Dict[str, Dict[VarIndex, Optional[float]]]

LPSolveStats = TypedDict(
    "LPSolveStats",
    {
        "LP log": str,
        "LP value": Optional[float],
    },
)

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
    },
    total=False,
)

IterationCallback = Callable[[], bool]

LazyCallback = Callable[[Any, Any], None]

UserCutCallback = Callable[["InternalSolver", Any], None]

SolverParams = Dict[str, Any]

BranchPriorities = Solution


class Constraint:
    pass
