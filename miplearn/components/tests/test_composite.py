#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock, call

from nltk import Model

from miplearn import Component, LearningSolver, Instance
from miplearn.components.composite import CompositeComponent


def test_composite():
    solver, instance, model = (
        Mock(spec=LearningSolver),
        Mock(spec=Instance),
        Mock(spec=Model),
    )

    c1 = Mock(spec=Component)
    c2 = Mock(spec=Component)
    cc = CompositeComponent([c1, c2])

    # Should broadcast before_solve
    cc.before_solve(solver, instance, model)
    c1.before_solve.assert_has_calls([call(solver, instance, model)])
    c2.before_solve.assert_has_calls([call(solver, instance, model)])

    # Should broadcast after_solve
    cc.after_solve(solver, instance, model, {}, {})
    c1.after_solve.assert_has_calls([call(solver, instance, model, {}, {})])
    c2.after_solve.assert_has_calls([call(solver, instance, model, {}, {})])

    # Should broadcast fit
    cc.fit([1, 2, 3])
    c1.fit.assert_has_calls([call([1, 2, 3])])
    c2.fit.assert_has_calls([call([1, 2, 3])])

    # Should broadcast lazy_cb
    cc.lazy_cb(solver, instance, model)
    c1.lazy_cb.assert_has_calls([call(solver, instance, model)])
    c2.lazy_cb.assert_has_calls([call(solver, instance, model)])

    # Should broadcast iteration_cb
    cc.iteration_cb(solver, instance, model)
    c1.iteration_cb.assert_has_calls([call(solver, instance, model)])
    c2.iteration_cb.assert_has_calls([call(solver, instance, model)])

    # If at least one child component returns true, iteration_cb should return True
    c1.iteration_cb = Mock(return_value=True)
    c2.iteration_cb = Mock(return_value=False)
    assert cc.iteration_cb(solver, instance, model)

    # If all children return False, iteration_cb should return False
    c1.iteration_cb = Mock(return_value=False)
    c2.iteration_cb = Mock(return_value=False)
    assert not cc.iteration_cb(solver, instance, model)
