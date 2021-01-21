#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from pyomo import environ as pe
from scipy.stats import randint

from miplearn.solvers.pyomo.base import BasePyomoSolver


class CplexPyomoSolver(BasePyomoSolver):
    """
    An InternalSolver that uses CPLEX and the Pyomo modeling language.

    Parameters
    ----------
    params: dict
        Dictionary of options to pass to the Pyomo solver. For example,
        {"mip_display": 5} to increase the log verbosity.
    """

    def __init__(self, params=None):
        super().__init__(
            solver_factory=pe.SolverFactory("cplex_persistent"),
            params={
                "randomseed": randint(low=0, high=1000).rvs(),
                "mip_display": 4,
            },
        )

    def _get_warm_start_regexp(self):
        return "MIP start .* with objective ([0-9.e+-]*)\\."

    def _get_node_count_regexp(self):
        return "^[ *] *([0-9]+)"
