#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Optional, Dict, Callable, Any, Union, Tuple, List, Set, Hashable

from mypy_extensions import TypedDict

VarIndex = Union[str, int, Tuple[Union[str, int]]]

Solution = Dict[str, Dict[VarIndex, Optional[float]]]

TrainingSample = TypedDict(
    "TrainingSample",
    {
        "LP log": str,
        "LP solution": Optional[Solution],
        "LP value": Optional[float],
        "LazyStatic: All": Set[str],
        "LazyStatic: Enforced": Set[str],
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

InstanceFeatures = TypedDict(
    "InstanceFeatures",
    {
        "User features": List[float],
        "Lazy constraint count": int,
    },
    total=False,
)

VariableFeatures = TypedDict(
    "VariableFeatures",
    {
        "Category": Optional[Hashable],
        "User features": Optional[List[float]],
    },
    total=False,
)

ConstraintFeatures = TypedDict(
    "ConstraintFeatures",
    {
        "RHS": float,
        "LHS": Dict[str, float],
        "Sense": str,
        "Category": Optional[Hashable],
        "User features": Optional[List[float]],
        "Lazy": bool,
    },
    total=False,
)

Features = TypedDict(
    "Features",
    {
        "Instance": InstanceFeatures,
        "Variables": Dict[str, Dict[VarIndex, VariableFeatures]],
        "Constraints": Dict[str, ConstraintFeatures],
    },
    total=False,
)

IterationCallback = Callable[[], bool]

LazyCallback = Callable[[Any, Any], None]

SolverParams = Dict[str, Any]

BranchPriorities = Solution


class Constraint:
    pass
