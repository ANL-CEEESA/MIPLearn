#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Optional

from overrides import overrides
from pyomo import environ as pe
from scipy.stats import randint

from miplearn.solvers.pyomo.base import BasePyomoSolver
from miplearn.types import SolverParams


class CplexPyomoSolver(BasePyomoSolver):
    """
    An InternalSolver that uses CPLEX and the Pyomo modeling language.

    Parameters
    ----------
    params: dict
        Dictionary of options to pass to the Pyomo solver. For example,
        {"mip_display": 5} to increase the log verbosity.
    """

    def __init__(
        self,
        params: Optional[SolverParams] = None,
    ) -> None:
        if params is None:
            params = {}
        params["randomseed"] = randint(low=0, high=1000).rvs()
        if "mip_display" not in params.keys():
            params["mip_display"] = 4
        super().__init__(
            solver_factory=pe.SolverFactory("cplex_persistent"),
            params=params,
        )

    @overrides
    def _get_warm_start_regexp(self):
        return "MIP start .* with objective ([0-9.e+-]*)\\."

    @overrides
    def _get_node_count_regexp(self):
        return "^[ *] *([0-9]+)"

    @overrides
    def clone(self) -> "CplexPyomoSolver":
        return CplexPyomoSolver(params=self.params)
