#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from typing import List

import pytest

from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.pyomo.gurobi import GurobiPyomoSolver
from miplearn.solvers.pyomo.xpress import XpressPyomoSolver
from miplearn.solvers.tests import run_internal_solver_tests

logger = logging.getLogger(__name__)


@pytest.fixture
def internal_solvers() -> List[InternalSolver]:
    return [
        XpressPyomoSolver(),
        GurobiSolver(),
        GurobiPyomoSolver(),
    ]


def test_xpress_pyomo_solver() -> None:
    run_internal_solver_tests(XpressPyomoSolver())


def test_gurobi_pyomo_solver() -> None:
    run_internal_solver_tests(GurobiPyomoSolver())


def test_gurobi_solver() -> None:
    run_internal_solver_tests(GurobiSolver())


def test_redundancy() -> None:
    solver = GurobiSolver()
    instance = solver.build_test_instance_redundancy()
    solver.set_instance(instance)
    stats = solver.solve_lp()
    assert stats.lp_value == 1.0
    constraints = solver.get_constraints()
    assert constraints.names[0] == "c1"
    assert constraints.slacks[0] == 0.0

    solver.relax_constraints(["c1"])
    stats = solver.solve_lp()
    assert stats.lp_value == 2.0
    assert solver.is_constraint_satisfied(["c1"]) == [False]

    solver.enforce_constraints(["c1"])
    stats = solver.solve_lp()
    assert stats.lp_value == 1.0
