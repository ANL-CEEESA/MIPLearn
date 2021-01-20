#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import sys
import logging
from io import StringIO
from pyomo import environ as pe
from scipy.stats import randint

from .base import BasePyomoSolver
from .. import RedirectOutput

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

    def __init__(self, params=None):
        super().__init__(
            solver_factory=pe.SolverFactory("xpress_persistent"),
            params={
                "randomseed": randint(low=0, high=1000).rvs(),
            },
        )
