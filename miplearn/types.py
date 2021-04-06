#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Optional, Dict, Callable, Any, Union, Tuple, List, Set, Hashable
from dataclasses import dataclass

from mypy_extensions import TypedDict

VarIndex = Union[str, int, Tuple[Union[str, int]]]

Solution = Dict[str, Dict[VarIndex, Optional[float]]]


@dataclass
class TrainingSample:
    lp_log: Optional[str] = None
    lp_solution: Optional[Solution] = None
    lp_value: Optional[float] = None
    lazy_enforced: Optional[Set[str]] = None
    lower_bound: Optional[float] = None
    mip_log: Optional[str] = None
    solution: Optional[Solution] = None
    upper_bound: Optional[float] = None
    slacks: Optional[Dict[str, float]] = None


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


@dataclass
class InstanceFeatures:
    user_features: Optional[List[float]] = None
    lazy_constraint_count: int = 0


@dataclass
class VariableFeatures:
    category: Optional[Hashable] = None
    user_features: Optional[List[float]] = None


@dataclass
class ConstraintFeatures:
    rhs: Optional[float] = None
    lhs: Optional[Dict[str, float]] = None
    sense: Optional[str] = None
    category: Optional[Hashable] = None
    user_features: Optional[List[float]] = None
    lazy: bool = False


@dataclass
class Features:
    instance: Optional[InstanceFeatures] = None
    variables: Optional[Dict[str, Dict[VarIndex, VariableFeatures]]] = None
    constraints: Optional[Dict[str, ConstraintFeatures]] = None


IterationCallback = Callable[[], bool]

LazyCallback = Callable[[Any, Any], None]

SolverParams = Dict[str, Any]

BranchPriorities = Solution


class Constraint:
    pass
