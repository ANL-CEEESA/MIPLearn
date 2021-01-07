from miplearn import LearningSolver, GurobiSolver
from miplearn.components.steps.convert_tight import ConvertTightIneqsIntoEqsStep
from miplearn.components.steps.relax_integrality import RelaxIntegralityStep
from miplearn.problems.knapsack import GurobiKnapsackInstance


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
