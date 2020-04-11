#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import pickle
import tempfile

import pyomo.environ as pe
from miplearn import LearningSolver, BranchPriorityComponent
from miplearn.problems.knapsack import KnapsackInstance
from miplearn.solvers import GurobiSolver, CPLEXSolver


def _get_instance():
    return KnapsackInstance(
        weights=[23., 26., 20., 18.],
        prices=[505., 352., 458., 220.],
        capacity=67.,
    )


def test_internal_solver():
    for solver in [GurobiSolver(), CPLEXSolver(presolve=False)]:
        instance = _get_instance()
        model = instance.to_model()

        solver.set_instance(instance, model)
        solver.set_warm_start({
            "x": {
                0: 1.0,
                1: 0.0,
                2: 1.0,
                3: 1.0,
            }
        })

        stats = solver.solve()
        assert stats["Lower bound"] == 1183.0
        assert stats["Upper bound"] == 1183.0
        assert stats["Sense"] == "max"
        assert isinstance(stats["Wallclock time"], float)
        assert isinstance(stats["Nodes"], int)

        solution = solver.get_solution()
        assert solution["x"][0] == 1.0
        assert solution["x"][1] == 0.0
        assert solution["x"][2] == 1.0
        assert solution["x"][3] == 1.0

        stats = solver.solve_lp()
        assert round(stats["Optimal value"], 3) == 1287.923

        solution = solver.get_solution()
        assert round(solution["x"][0], 3) == 1.000
        assert round(solution["x"][1], 3) == 0.923
        assert round(solution["x"][2], 3) == 1.000
        assert round(solution["x"][3], 3) == 0.000

        model.cut = pe.Constraint(expr=model.x[0] <= 0.5)
        solver.add_constraint(model.cut)
        solver.solve_lp()
        assert model.x[0].value == 0.5


def test_learning_solver():
    instance = _get_instance()
    for mode in ["exact", "heuristic"]:
        for internal_solver in ["cplex", "gurobi", GurobiSolver]:
            solver = LearningSolver(time_limit=300,
                                    gap_tolerance=1e-3,
                                    threads=1,
                                    solver=internal_solver,
                                    mode=mode)

            solver.solve(instance)
            assert instance.solution["x"][0] == 1.0
            assert instance.solution["x"][1] == 0.0
            assert instance.solution["x"][2] == 1.0
            assert instance.solution["x"][3] == 1.0
            assert instance.lower_bound == 1183.0
            assert instance.upper_bound == 1183.0

            assert round(instance.lp_solution["x"][0], 3) == 1.000
            assert round(instance.lp_solution["x"][1], 3) == 0.923
            assert round(instance.lp_solution["x"][2], 3) == 1.000
            assert round(instance.lp_solution["x"][3], 3) == 0.000
            assert round(instance.lp_value, 3) == 1287.923

            solver.fit([instance])
            solver.solve(instance)

            # Assert solver is picklable
            with tempfile.TemporaryFile() as file:
                pickle.dump(solver, file)


def test_parallel_solve():
    instances = [_get_instance() for _ in range(10)]
    solver = LearningSolver()
    results = solver.parallel_solve(instances, n_jobs=3)
    assert len(results) == 10
    for instance in instances:
        assert len(instance.solution["x"].keys()) == 4


def test_add_components():
    solver = LearningSolver(components=[])
    solver.add(BranchPriorityComponent())
    solver.add(BranchPriorityComponent())
    assert len(solver.components) == 1
    assert "BranchPriorityComponent" in solver.components
