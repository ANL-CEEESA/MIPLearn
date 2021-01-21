#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging

from pyomo import environ as pe
from scipy.stats import randint

from miplearn.solvers.pyomo.base import BasePyomoSolver
from miplearn.types import SolverParams

logger = logging.getLogger(__name__)


class XpressPyomoSolver(BasePyomoSolver):
    """
    An InternalSolver that uses XPRESS and the Pyomo modeling language.

    Parameters
    ----------
    params: dict
        Dictionary of options to pass to the Pyomo solver. For example,
        {"Threads": 4} to set the number of threads.
    """

    def __init__(self, params: SolverParams = None) -> None:
        if params is None:
            params = {}
        if "randomseed" not in params.keys():
            params["randomseed"] = randint(low=0, high=1000).rvs()
        super().__init__(
            solver_factory=pe.SolverFactory("xpress_persistent"),
            params=params,
        )
