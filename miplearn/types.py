#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Optional, Dict, Callable, Any, Union, Tuple

from mypy_extensions import TypedDict

VarIndex = Union[str, int, Tuple[Union[str, int]]]

Solution = Dict[str, Dict[VarIndex, Optional[float]]]

TrainingSample = TypedDict(
    "TrainingSample",
    {
        "LP log": str,
        "LP solution": Optional[Solution],
        "LP value": Optional[float],
        "Lower bound": Optional[float],
        "MIP log": str,
        "Solution": Optional[Solution],
        "Upper bound": Optional[float],
        "slacks": Dict,
    },
    total=False,
)

LPSolveStats = TypedDict(
    "LPSolveStats",
    {
        "Optimal value": Optional[float],
        "Log": str,
    },
)

MIPSolveStats = TypedDict(
    "MIPSolveStats",
    {
        "Lower bound": Optional[float],
        "Upper bound": Optional[float],
        "Wallclock time": float,
        "Nodes": Optional[int],
        "Sense": str,
        "Log": str,
        "Warm start value": Optional[float],
        "LP value": Optional[float],
    },
)

LearningSolveStats = TypedDict(
    "LearningSolveStats",
    {
        "Gap": Optional[float],
        "Instance": Union[str, int],
        "LP value": Optional[float],
        "Log": str,
        "Lower bound": Optional[float],
        "Mode": str,
        "Nodes": Optional[int],
        "Sense": str,
        "Solver": str,
        "Upper bound": Optional[float],
        "Wallclock time": float,
        "Warm start value": Optional[float],
        "Primal: free": int,
        "Primal: zero": int,
        "Primal: one": int,
        "Objective: predicted LB": float,
        "Objective: predicted UB": float,
    },
    total=False,
)

IterationCallback = Callable[[], bool]

LazyCallback = Callable[[Any, Any], None]

SolverParams = Dict[str, Any]

BranchPriorities = Solution


class Constraint:
    pass
