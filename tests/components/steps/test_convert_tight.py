#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from unittest.mock import Mock

from miplearn.classifiers import Classifier
from miplearn.components.steps.convert_tight import ConvertTightIneqsIntoEqsStep
from miplearn.components.steps.relax_integrality import RelaxIntegralityStep
from miplearn.instance.base import Instance
from miplearn.problems.knapsack import GurobiKnapsackInstance
from miplearn.solvers.gurobi import GurobiSolver
from miplearn.solvers.learning import LearningSolver


def test_convert_tight_usage():
    instance = GurobiKnapsackInstance(
        weights=[3.0, 5.0, 10.0],
        prices=[1.0, 1.0, 1.0],
        capacity=16.0,
    )
    solver = LearningSolver(
        solver=GurobiSolver,
        components=[
            RelaxIntegralityStep(),
            ConvertTightIneqsIntoEqsStep(),
        ],
    )

    # Solve original problem
    stats = solver.solve(instance)
    original_upper_bound = stats["Upper bound"]

    # Should collect training data
    assert instance.training_data[0].slacks["eq_capacity"] == 0.0

    # Fit and resolve
    solver.fit([instance])
    stats = solver.solve(instance)

    # Objective value should be the same
    assert stats["Upper bound"] == original_upper_bound
    assert stats["ConvertTight: Inf iterations"] == 0
    assert stats["ConvertTight: Subopt iterations"] == 0


class SampleInstance(Instance):
    def to_model(self):
        import gurobipy as grb

        m = grb.Model("model")
        x1 = m.addVar(name="x1")
        x2 = m.addVar(name="x2")
        m.setObjective(x1 + 2 * x2, grb.GRB.MAXIMIZE)
        m.addConstr(x1 <= 2, name="c1")
        m.addConstr(x2 <= 2, name="c2")
        m.addConstr(x1 + x2 <= 3, name="c2")
        return m


def test_convert_tight_infeasibility():
    comp = ConvertTightIneqsIntoEqsStep()
    comp.classifiers = {
        "c1": Mock(spec=Classifier),
        "c2": Mock(spec=Classifier),
        "c3": Mock(spec=Classifier),
    }
    comp.classifiers["c1"].predict_proba = Mock(return_value=[[0, 1]])
    comp.classifiers["c2"].predict_proba = Mock(return_value=[[0, 1]])
    comp.classifiers["c3"].predict_proba = Mock(return_value=[[1, 0]])

    solver = LearningSolver(
        solver=GurobiSolver,
        components=[comp],
        solve_lp=False,
    )
    instance = SampleInstance()
    stats = solver.solve(instance)
    assert stats["Upper bound"] == 5.0
    assert stats["ConvertTight: Inf iterations"] == 1
    assert stats["ConvertTight: Subopt iterations"] == 0


def test_convert_tight_suboptimality():
    comp = ConvertTightIneqsIntoEqsStep(check_optimality=True)
    comp.classifiers = {
        "c1": Mock(spec=Classifier),
        "c2": Mock(spec=Classifier),
        "c3": Mock(spec=Classifier),
    }
    comp.classifiers["c1"].predict_proba = Mock(return_value=[[0, 1]])
    comp.classifiers["c2"].predict_proba = Mock(return_value=[[1, 0]])
    comp.classifiers["c3"].predict_proba = Mock(return_value=[[0, 1]])

    solver = LearningSolver(
        solver=GurobiSolver,
        components=[comp],
        solve_lp=False,
    )
    instance = SampleInstance()
    stats = solver.solve(instance)
    assert stats["Upper bound"] == 5.0
    assert stats["ConvertTight: Inf iterations"] == 0
    assert stats["ConvertTight: Subopt iterations"] == 1


def test_convert_tight_optimal():
    comp = ConvertTightIneqsIntoEqsStep()
    comp.classifiers = {
        "c1": Mock(spec=Classifier),
        "c2": Mock(spec=Classifier),
        "c3": Mock(spec=Classifier),
    }
    comp.classifiers["c1"].predict_proba = Mock(return_value=[[1, 0]])
    comp.classifiers["c2"].predict_proba = Mock(return_value=[[0, 1]])
    comp.classifiers["c3"].predict_proba = Mock(return_value=[[0, 1]])

    solver = LearningSolver(
        solver=GurobiSolver,
        components=[comp],
        solve_lp=False,
    )
    instance = SampleInstance()
    stats = solver.solve(instance)
    assert stats["Upper bound"] == 5.0
    assert stats["ConvertTight: Inf iterations"] == 0
    assert stats["ConvertTight: Subopt iterations"] == 0
