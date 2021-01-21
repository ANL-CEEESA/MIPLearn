#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Optional, Dict, Callable, Any, Union, List

from mypy_extensions import TypedDict

TrainingSample = TypedDict(
    "TrainingSample",
    {
        "LP log": str,
        "LP solution": Optional[Dict],
        "LP value": Optional[float],
        "Lower bound": Optional[float],
        "MIP log": str,
        "Solution": Optional[Dict],
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

IterationCallback = Callable[[], bool]

LazyCallback = Callable[[Any, Any], None]

VarIndex = Union[str, int, List[Union[str, int]]]
