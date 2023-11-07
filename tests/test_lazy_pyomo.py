#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
from typing import Any, Hashable, List

import pyomo.environ as pe

from miplearn.solvers.pyomo import PyomoModel

logger = logging.getLogger(__name__)


def _build_model() -> PyomoModel:
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(0, 5), domain=pe.Integers)
    m.obj = pe.Objective(expr=-m.x)
    m.cons = pe.ConstraintList()

    def lazy_separate(model: PyomoModel) -> List[Hashable]:
        model.solver.cbGetSolution(vars=[m.x])
        if m.x.value > 0.5:
            return [m.x.value]
        else:
            return []

    def lazy_enforce(model: PyomoModel, violations: List[Any]) -> None:
        for v in violations:
            model.add_constr(m.cons.add(m.x <= round(v - 1)))

    return PyomoModel(
        m,
        "gurobi_persistent",
        lazy_separate=lazy_separate,
        lazy_enforce=lazy_enforce,
    )


def test_pyomo_callback() -> None:
    model = _build_model()
    model.optimize()
    assert model.lazy_ is not None
    assert len(model.lazy_) > 0
    assert model.inner.x.value == 0.0
