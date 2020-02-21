# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright © 2020, UChicago Argonne, LLC. All rights reserved.
# Released under the modified BSD license. See COPYING.md for more details.
# Written by Alinson S. Xavier <axavier@anl.gov>

from .solvers import LearningSolver
from copy import deepcopy
import pandas as pd
from tqdm.auto import tqdm

class BenchmarkRunner:
    def __init__(self, solvers):
        assert isinstance(solvers, dict)
        for solver in solvers.values():
            assert isinstance(solver, LearningSolver)
        self.solvers = solvers
        self.results = None
        
    def solve(self, instances, fit=True, tee=False):
        for (name, solver) in self.solvers.items():
            for i in tqdm(range(len((instances)))):
                results = solver.solve(deepcopy(instances[i]), tee=tee)
                self._push_result(results, solver=solver, name=name, instance=i)
                if fit:
                    solver.fit()

    def parallel_solve(self, instances, n_jobs=1, n_trials=1):
        instances = instances * n_trials
        for (name, solver) in self.solvers.items():
            results = solver.parallel_solve(instances,
                                            n_jobs=n_jobs,
                                            label="Solve (%s)" % name,
                                            collect_training_data=False)
            for i in range(len(instances)):
                self._push_result(results[i], solver=solver, name=name, instance=i)
    
    def raw_results(self):
        return self.results
    
    def save_results(self, filename):
        self.results.to_csv(filename)
        
    def load_results(self, filename):
        self.results = pd.read_csv(filename, index_col=0)
        
    def load_state(self, filename):
        for (name, solver) in self.solvers.items():
            solver.load_state(filename)

    def fit(self):
        for (name, solver) in self.solvers.items():
            solver.fit()
            
    def _push_result(self, result, solver, name, instance):
        if self.results is None:
            self.results = pd.DataFrame(columns=["Solver",
                                                 "Instance",
                                                 "Wallclock Time",
                                                 "Lower Bound",
                                                 "Upper Bound",
                                                 "Gap",
                                                 "Nodes",
                                                 "Mode",
                                                ])
        wallclock_time = None
        for key in ["Time", "Wall time", "Wallclock time"]:
            if key not in result["Solver"][0].keys():
                continue
            if str(result["Solver"][0][key]) == "<undefined>":
                continue
            wallclock_time = float(result["Solver"][0][key])
        nodes = result["Solver"][0]["Nodes"]
        lb = result["Problem"][0]["Lower bound"]
        ub = result["Problem"][0]["Upper bound"]
        gap = (ub - lb) / lb
        self.results = self.results.append({
            "Solver": name,
            "Instance": instance,
            "Wallclock Time": wallclock_time,
            "Lower Bound": lb,
            "Upper Bound": ub,
            "Gap": gap,
            "Nodes": nodes,
            "Mode": solver.mode,
        }, ignore_index=True)
        groups = self.results.groupby("Instance")
        best_lower_bound = groups["Lower Bound"].transform("max")
        best_upper_bound = groups["Upper Bound"].transform("min")
        best_gap = groups["Gap"].transform("min")
        best_nodes = groups["Nodes"].transform("min")
        best_wallclock_time = groups["Wallclock Time"].transform("min")
        self.results["Relative Lower Bound"] = \
                self.results["Lower Bound"] / best_lower_bound
        self.results["Relative Upper Bound"] = \
                self.results["Upper Bound"] / best_upper_bound
        self.results["Relative Wallclock Time"] = \
                self.results["Wallclock Time"] / best_wallclock_time
        self.results["Relative Gap"] = \
                self.results["Gap"] / best_gap
        self.results["Relative Nodes"] = \
                self.results["Nodes"] / best_nodes
