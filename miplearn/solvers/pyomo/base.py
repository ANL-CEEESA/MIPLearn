#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import re
import sys
import logging
import pyomo
from abc import abstractmethod
from io import StringIO
from pyomo import environ as pe
from pyomo.core import Var, Constraint

from .. import RedirectOutput
from ..internal import InternalSolver
from ...instance import Instance

logger = logging.getLogger(__name__)


class BasePyomoSolver(InternalSolver):
    """
    Base class for all Pyomo solvers.
    """

    def __init__(self):
        self.instance = None
        self.model = None
        self._all_vars = None
        self._bin_vars = None
        self._is_warm_start_available = False
        self._pyomo_solver = None
        self._obj_sense = None
        self._varname_to_var = {}
        self._cname_to_constr = {}

    def solve_lp(self, tee=False):
        for var in self._bin_vars:
            lb, ub = var.bounds
            var.setlb(lb)
            var.setub(ub)
            var.domain = pyomo.core.base.set_types.Reals
            self._pyomo_solver.update_var(var)
        results = self._pyomo_solver.solve(tee=tee)
        for var in self._bin_vars:
            var.domain = pyomo.core.base.set_types.Binary
            self._pyomo_solver.update_var(var)
        return {
            "Optimal value": results["Problem"][0]["Lower bound"],
        }

    def get_solution(self):
        solution = {}
        for var in self.model.component_objects(Var):
            solution[str(var)] = {}
            for index in var:
                solution[str(var)][index] = var[index].value
        return solution

    def get_value(self, var_name, index):
        var = self._varname_to_var[var_name]
        return var[index].value

    def get_variables(self):
        variables = {}
        for var in self.model.component_objects(Var):
            variables[str(var)] = []
            for index in var:
                variables[str(var)] += [index]
        return variables

    def set_warm_start(self, solution):
        self.clear_warm_start()
        count_total, count_fixed = 0, 0
        for var_name in solution:
            var = self._varname_to_var[var_name]
            for index in solution[var_name]:
                count_total += 1
                var[index].value = solution[var_name][index]
                if solution[var_name][index] is not None:
                    count_fixed += 1
        if count_fixed > 0:
            self._is_warm_start_available = True
        logger.info(
            "Setting start values for %d variables (out of %d)"
            % (count_fixed, count_total)
        )

    def clear_warm_start(self):
        for var in self._all_vars:
            if not var.fixed:
                var.value = None
        self._is_warm_start_available = False

    def set_instance(self, instance, model=None):
        if model is None:
            model = instance.to_model()
        assert isinstance(instance, Instance)
        assert isinstance(model, pe.ConcreteModel)
        self.instance = instance
        self.model = model
        self._pyomo_solver.set_instance(model)
        self._update_obj()
        self._update_vars()
        self._update_constrs()

    def _update_obj(self):
        self._obj_sense = "max"
        if self._pyomo_solver._objective.sense == pyomo.core.kernel.objective.minimize:
            self._obj_sense = "min"

    def _update_vars(self):
        self._all_vars = []
        self._bin_vars = []
        self._varname_to_var = {}
        for var in self.model.component_objects(Var):
            self._varname_to_var[var.name] = var
            for idx in var:
                self._all_vars += [var[idx]]
                if var[idx].domain == pyomo.core.base.set_types.Binary:
                    self._bin_vars += [var[idx]]

    def _update_constrs(self):
        self._cname_to_constr = {}
        for constr in self.model.component_objects(Constraint):
            self._cname_to_constr[constr.name] = constr

    def fix(self, solution):
        count_total, count_fixed = 0, 0
        for varname in solution:
            for index in solution[varname]:
                var = self._varname_to_var[varname]
                count_total += 1
                if solution[varname][index] is None:
                    continue
                count_fixed += 1
                var[index].fix(solution[varname][index])
                self._pyomo_solver.update_var(var[index])
        logger.info(
            "Fixing values for %d variables (out of %d)"
            % (
                count_fixed,
                count_total,
            )
        )

    def add_constraint(self, constraint):
        self._pyomo_solver.add_constraint(constraint)
        self._update_constrs()

    def solve(self, tee=False, iteration_cb=None, lazy_cb=None):
        if lazy_cb is not None:
            raise Exception("lazy callback not supported")
        total_wallclock_time = 0
        streams = [StringIO()]
        if tee:
            streams += [sys.stdout]
        if iteration_cb is None:
            iteration_cb = lambda: False
        self.instance.found_violated_lazy_constraints = []
        self.instance.found_violated_user_cuts = []
        while True:
            logger.debug("Solving MIP...")
            with RedirectOutput(streams):
                results = self._pyomo_solver.solve(
                    tee=True,
                    warmstart=self._is_warm_start_available,
                )
            total_wallclock_time += results["Solver"][0]["Wallclock time"]
            should_repeat = iteration_cb()
            if not should_repeat:
                break
        log = streams[0].getvalue()
        return {
            "Lower bound": results["Problem"][0]["Lower bound"],
            "Upper bound": results["Problem"][0]["Upper bound"],
            "Wallclock time": total_wallclock_time,
            "Nodes": self._extract_node_count(log),
            "Sense": self._obj_sense,
            "Log": log,
            "Warm start value": self._extract_warm_start_value(log),
        }

    @staticmethod
    def __extract(log, regexp, default=None):
        value = default
        for line in log.splitlines():
            matches = re.findall(regexp, line)
            if len(matches) == 0:
                continue
            value = matches[0]
        return value

    def _extract_warm_start_value(self, log):
        value = self.__extract(log, self._get_warm_start_regexp())
        if value is not None:
            value = float(value)
        return value

    def _extract_node_count(self, log):
        return int(self.__extract(log, self._get_node_count_regexp(), default=1))

    def set_threads(self, threads):
        key = self._get_threads_option_name()
        self._pyomo_solver.options[key] = threads

    def set_time_limit(self, time_limit):
        key = self._get_time_limit_option_name()
        self._pyomo_solver.options[key] = time_limit

    def set_node_limit(self, node_limit):
        key = self._get_node_limit_option_name()
        self._pyomo_solver.options[key] = node_limit

    def set_gap_tolerance(self, gap_tolerance):
        key = self._get_gap_tolerance_option_name()
        self._pyomo_solver.options[key] = gap_tolerance

    def get_constraint_ids(self):
        return list(self._cname_to_constr.keys())

    def extract_constraint(self, cid):
        raise Exception("Not implemented")

    def is_constraint_satisfied(self, cobj):
        raise Exception("Not implemented")

    @abstractmethod
    def _get_warm_start_regexp(self):
        pass

    @abstractmethod
    def _get_node_count_regexp(self):
        pass

    @abstractmethod
    def _get_threads_option_name(self):
        pass

    @abstractmethod
    def _get_time_limit_option_name(self):
        pass

    @abstractmethod
    def _get_node_limit_option_name(self):
        pass

    @abstractmethod
    def _get_gap_tolerance_option_name(self):
        pass

    def relax(self):
        raise Exception("not implemented")

    def get_constraint_slacks(self):
        raise Exception("not implemented")
