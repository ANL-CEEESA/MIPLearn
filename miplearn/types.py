#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from typing import Optional, Dict, Callable, Any, Union, TYPE_CHECKING

from mypy_extensions import TypedDict

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from miplearn.solvers.learning import InternalSolver

Category = str
IterationCallback = Callable[[], bool]
LazyCallback = Callable[[Any, Any], None]
SolverParams = Dict[str, Any]
UserCutCallback = Callable[["InternalSolver", Any], None]
VariableName = str
Solution = Dict[VariableName, Optional[float]]

LearningSolveStats = TypedDict(
    "LearningSolveStats",
    {
        "Gap": Optional[float],
        "Instance": Union[str, int],
        "lp_log": str,
        "lp_value": Optional[float],
        "lp_wallclock_time": Optional[float],
        "mip_lower_bound": Optional[float],
        "mip_log": str,
        "Mode": str,
        "mip_nodes": Optional[int],
        "Objective: Predicted lower bound": float,
        "Objective: Predicted upper bound": float,
        "Primal: Free": int,
        "Primal: One": int,
        "Primal: Zero": int,
        "Sense": str,
        "Solver": str,
        "mip_upper_bound": Optional[float],
        "mip_wallclock_time": float,
        "mip_warm_start_value": Optional[float],
        "LazyStatic: Removed": int,
        "LazyStatic: Kept": int,
        "LazyStatic: Restored": int,
        "LazyStatic: Iterations": int,
        "UserCuts: Added ahead-of-time": int,
        "UserCuts: Added in callback": int,
    },
    total=False,
)
