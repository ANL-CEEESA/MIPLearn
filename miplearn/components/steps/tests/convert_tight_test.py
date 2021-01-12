from miplearn import LearningSolver, GurobiSolver, Instance, Classifier
from miplearn.components.steps.convert_tight import ConvertTightIneqsIntoEqsStep
from miplearn.components.steps.relax_integrality import RelaxIntegralityStep
from miplearn.problems.knapsack import GurobiKnapsackInstance

from unittest.mock import Mock


def test_convert_tight_usage():
    instance = GurobiKnapsackInstance(
        weights=[3.0, 5.0, 10.0],
        prices=[1.0, 1.0, 1.0],
        capacity=16.0,
    )
    solver = LearningSolver(
        solver=GurobiSolver(),
        components=[
            RelaxIntegralityStep(),
            ConvertTightIneqsIntoEqsStep(),
        ],
    )

    # Solve original problem
    solver.solve(instance)
    original_upper_bound = instance.upper_bound

    # Should collect training data
    assert hasattr(instance, "slacks")
    assert instance.slacks["eq_capacity"] == 0.0

    # Fit and resolve
    solver.fit([instance])
    solver.solve(instance)

    # Objective value should be the same
    assert instance.upper_bound == original_upper_bound


class TestInstance(Instance):
    def to_model(self):
        import gurobipy as grb
        from gurobipy import GRB

        m = grb.Model("model")
        x1 = m.addVar(name="x1")
        x2 = m.addVar(name="x2")
        m.setObjective(x1 + 2 * x2, grb.GRB.MAXIMIZE)
        m.addConstr(x1 <= 2, name="c1")
        m.addConstr(x2 <= 2, name="c2")
        m.addConstr(x1 + x2 <= 3, name="c2")
        return m


def test_convert_tight_infeasibility():
    comp = ConvertTightIneqsIntoEqsStep(
        check_converted=True,
    )
    comp.classifiers = {
        "c1": Mock(spec=Classifier),
        "c2": Mock(spec=Classifier),
        "c3": Mock(spec=Classifier),
    }
    comp.classifiers["c1"].predict_proba = Mock(return_value=[[0, 1]])
    comp.classifiers["c2"].predict_proba = Mock(return_value=[[0, 1]])
    comp.classifiers["c3"].predict_proba = Mock(return_value=[[1, 0]])

    solver = LearningSolver(
        solver=GurobiSolver(params={}),
        components=[comp],
        solve_lp_first=False,
    )
    instance = TestInstance()
    solver.solve(instance)
    assert instance.lower_bound == 5.0