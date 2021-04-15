#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import Optional

from overrides import overrides
from pyomo import environ as pe
from scipy.stats import randint

from miplearn.solvers.pyomo.base import BasePyomoSolver
from miplearn.types import SolverParams, BranchPriorities

logger = logging.getLogger(__name__)


class GurobiPyomoSolver(BasePyomoSolver):
    """
    An InternalSolver that uses Gurobi and the Pyomo modeling language.

    Parameters
    ----------
    params: dict
        Dictionary of options to pass to the Pyomo solver. For example,
        {"Threads": 4} to set the number of threads.
    """

    def __init__(
        self,
        params: Optional[SolverParams] = None,
    ) -> None:
        if params is None:
            params = {}
        params["seed"] = randint(low=0, high=1000).rvs()
        super().__init__(
            solver_factory=pe.SolverFactory("gurobi_persistent"),
            params=params,
        )

    @overrides
    def clone(self) -> "GurobiPyomoSolver":
        return GurobiPyomoSolver(params=self.params)

    @overrides
    def set_branching_priorities(self, priorities: BranchPriorities) -> None:
        from gurobipy import GRB

        for (varname, priority) in priorities.items():
            if priority is None:
                continue
            var = self._varname_to_var[varname]
            gvar = self._pyomo_solver._pyomo_var_to_solver_var_map[var]
            gvar.setAttr(GRB.Attr.BranchPriority, int(round(priority)))

    @overrides
    def _extract_node_count(self, log: str) -> int:
        return max(1, int(self._pyomo_solver._solver_model.getAttr("NodeCount")))

    @overrides
    def _get_warm_start_regexp(self) -> str:
        return "MIP start with objective ([0-9.e+-]*)"

    @overrides
    def _get_node_count_regexp(self) -> Optional[str]:
        return None
