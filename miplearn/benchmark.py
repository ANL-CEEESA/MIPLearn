# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

from .solvers import LearningSolver
import pandas as pd

class BenchmarkRunner:
    def __init__(self, solvers):
        assert isinstance(solvers, dict)
        for solver in solvers.values():
            assert isinstance(solver, LearningSolver)
        self.solvers = solvers
        self.results = None
        
    def load_fit(self, filename):
        for (name, solver) in self.solvers.items():
            solver.load(filename)
            solver.fit()
            
    def parallel_solve(self, instances, n_jobs=1):
        if self.results is None:
            self.results = pd.DataFrame(columns=["Solver",
                                                 "Instance",
                                                 "Wallclock Time",
                                                 "Obj Value",
                                                ])
        for (name, solver) in self.solvers.items():
            results = solver.parallel_solve(instances, n_jobs=n_jobs, label=name)
            for i in range(len(instances)):
                wallclock_time = None
                for key in ["Time", "Wall time", "Wallclock time"]:
                    if key not in results[i]["Solver"][0].keys():
                        continue
                    if str(results[i]["Solver"][0][key]) == "<undefined>":
                        continue
                    wallclock_time = float(results[i]["Solver"][0][key])
                self.results = self.results.append({
                    "Solver": name,
                    "Instance": i,
                    "Wallclock Time": wallclock_time,
                    "Obj Value": results[i]["Problem"][0]["Lower bound"]
                }, ignore_index=True)
                groups = self.results.groupby("Instance")
                best_obj_value = groups["Obj Value"].transform("max")
                best_wallclock_time = groups["Wallclock Time"].transform("min")
                self.results["Relative Obj Value"] = \
                        self.results["Obj Value"] / best_obj_value
                self.results["Relative Wallclock Time"] = \
                        self.results["Wallclock Time"] / best_wallclock_time
    
    def raw_results(self):
        return self.results
    
    def save_results(self, filename):
        self.results.to_csv(filename)
        
    def load_results(self, filename):
        self.results = pd.read_csv(filename, index_col=0)