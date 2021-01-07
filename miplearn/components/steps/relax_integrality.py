#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging

from miplearn import Component

logger = logging.getLogger(__name__)


class RelaxIntegralityStep(Component):
    """
    Component that relaxes all integrality constraints before the problem is solved.
    """

    def before_solve(self, solver, instance, _):
        logger.info("Relaxing integrality...")
        solver.internal_solver.relax()
