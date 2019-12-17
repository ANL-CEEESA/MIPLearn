# MIPLearn: A Machine-Learning Framework for Mixed-Integer Optimization
# Copyright (C) 2019-2020 Argonne National Laboratory. All rights reserved.
# Written by Alinson S. Xavier <axavier@anl.gov>

import pyomo.environ as pe

class LearningSolver:
    """
    LearningSolver is a Mixed-Integer Linear Programming (MIP) solver that uses information from
    previous runs to accelerate the solution of new, unseen instances.
    """
    
    def __init__(self):
        self.parent_solver = pe.SolverFactory('cplex_persistent')
        self.parent_solver.options["threads"] = 4
        
    def solve(self, params):
        """
        Solve the optimization problem represented by the given parameters.
        The parameters and the obtained solution is recorded.
        """
        model = params.to_model()
        self.parent_solver.set_instance(model)
        self.parent_solver.solve(tee=True)
    