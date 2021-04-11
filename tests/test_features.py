#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from miplearn.features import (
    FeaturesExtractor,
    InstanceFeatures,
    Variable,
    Constraint,
)
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.tests import assert_equals, _round_variables, _round_constraints

inf = float("inf")


def test_knapsack() -> None:
    solver = GurobiSolver()
    instance = solver.build_test_instance_knapsack()
    model = instance.to_model()
    solver.set_instance(instance, model)
    solver.solve_lp()

    FeaturesExtractor(solver).extract(instance)
    assert_equals(
        _round_variables(instance.features.variables),
        {
            "x[0]": Variable(
                basis_status="U",
                category="default",
                lower_bound=0.0,
                obj_coeff=505.0,
                reduced_cost=193.615385,
                sa_lb_down=-inf,
                sa_lb_up=1.0,
                sa_obj_down=311.384615,
                sa_obj_up=inf,
                sa_ub_down=0.913043,
                sa_ub_up=2.043478,
                type="C",
                upper_bound=1.0,
                user_features=[23.0, 505.0],
                value=1.0,
                alvarez_2017=[1.0, 0.32899, 0.0, 0.0, 1.0, 1.0, 5.265874, 0.0],
            ),
            "x[1]": Variable(
                basis_status="B",
                category="default",
                lower_bound=0.0,
                obj_coeff=352.0,
                reduced_cost=0.0,
                sa_lb_down=-inf,
                sa_lb_up=0.923077,
                sa_obj_down=317.777778,
                sa_obj_up=570.869565,
                sa_ub_down=0.923077,
                sa_ub_up=inf,
                type="C",
                upper_bound=1.0,
                user_features=[26.0, 352.0],
                value=0.923077,
                alvarez_2017=[
                    1.0,
                    0.229316,
                    0.0,
                    0.076923,
                    1.0,
                    1.0,
                    3.532875,
                    5.388476,
                ],
            ),
            "x[2]": Variable(
                basis_status="U",
                category="default",
                lower_bound=0.0,
                obj_coeff=458.0,
                reduced_cost=187.230769,
                sa_lb_down=-inf,
                sa_lb_up=1.0,
                sa_obj_down=270.769231,
                sa_obj_up=inf,
                sa_ub_down=0.9,
                sa_ub_up=2.2,
                type="C",
                upper_bound=1.0,
                user_features=[20.0, 458.0],
                value=1.0,
                alvarez_2017=[
                    1.0,
                    0.298371,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    5.232342,
                    0.0,
                ],
            ),
            "x[3]": Variable(
                basis_status="L",
                category="default",
                lower_bound=0.0,
                obj_coeff=220.0,
                reduced_cost=-23.692308,
                sa_lb_down=-0.111111,
                sa_lb_up=1.0,
                sa_obj_down=-inf,
                sa_obj_up=243.692308,
                sa_ub_down=0.0,
                sa_ub_up=inf,
                type="C",
                upper_bound=1.0,
                user_features=[18.0, 220.0],
                value=0.0,
                alvarez_2017=[
                    1.0,
                    0.143322,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    0.0,
                    3.16515,
                ],
            ),
        },
    )
    assert_equals(
        _round_constraints(instance.features.constraints),
        {
            "eq_capacity": Constraint(
                basis_status="N",
                category="eq_capacity",
                dual_value=13.538462,
                lazy=False,
                lhs={"x[0]": 23.0, "x[1]": 26.0, "x[2]": 20.0, "x[3]": 18.0},
                rhs=67.0,
                sa_rhs_down=43.0,
                sa_rhs_up=69.0,
                sense="<",
                slack=0.0,
                user_features=[0.0],
            )
        },
    )
    assert_equals(
        instance.features.instance,
        InstanceFeatures(
            user_features=[67.0, 21.75],
            lazy_constraint_count=0,
        ),
    )
