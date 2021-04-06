#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import re
import sys
from io import StringIO
from typing import Any, List, Dict, Optional

import pyomo
from pyomo import environ as pe
from pyomo.core import Var, Constraint
from pyomo.opt import TerminationCondition
from pyomo.opt.base.solvers import SolverFactory

from miplearn.instance import Instance
from miplearn.solvers import _RedirectOutput
from miplearn.solvers.internal import (
    InternalSolver,
    LPSolveStats,
    IterationCallback,
    LazyCallback,
    MIPSolveStats,
)
from miplearn.types import VarIndex, SolverParams, Solution, UserCutCallback

logger = logging.getLogger(__name__)


class BasePyomoSolver(InternalSolver):
    """
    Base class for all Pyomo solvers.
    """

    def __init__(
        self,
        solver_factory: SolverFactory,
        params: SolverParams,
    ) -> None:
        self.instance: Optional[Instance] = None
        self.model: Optional[pe.ConcreteModel] = None
        self._all_vars: List[pe.Var] = []
        self._bin_vars: List[pe.Var] = []
        self._is_warm_start_available: bool = False
        self._pyomo_solver: SolverFactory = solver_factory
        self._obj_sense: str = "min"
        self._varname_to_var: Dict[str, pe.Var] = {}
        self._cname_to_constr: Dict[str, pe.Constraint] = {}
        self._termination_condition: str = ""

        for (key, value) in params.items():
            self._pyomo_solver.options[key] = value

    def solve_lp(
        self,
        tee: bool = False,
    ) -> LPSolveStats:
        self.relax()
        streams: List[Any] = [StringIO()]
        if tee:
            streams += [sys.stdout]
        with _RedirectOutput(streams):
            results = self._pyomo_solver.solve(tee=True)
        self._restore_integrality()
        opt_value = None
        if not self.is_infeasible():
            opt_value = results["Problem"][0]["Lower bound"]
        return {
            "LP value": opt_value,
            "LP log": streams[0].getvalue(),
        }

    def _restore_integrality(self) -> None:
        for var in self._bin_vars:
            var.domain = pyomo.core.base.set_types.Binary
            self._pyomo_solver.update_var(var)

    def solve(
        self,
        tee: bool = False,
        iteration_cb: IterationCallback = None,
        lazy_cb: LazyCallback = None,
        user_cut_cb: UserCutCallback = None,
    ) -> MIPSolveStats:
        if lazy_cb is not None:
            raise Exception("lazy callback not currently supported")
        if user_cut_cb is not None:
            raise Exception("user cut callback not currently supported")
        total_wallclock_time = 0
        streams: List[Any] = [StringIO()]
        if tee:
            streams += [sys.stdout]
        if iteration_cb is None:
            iteration_cb = lambda: False
        while True:
            logger.debug("Solving MIP...")
            with _RedirectOutput(streams):
                results = self._pyomo_solver.solve(
                    tee=True,
                    warmstart=self._is_warm_start_available,
                )
            total_wallclock_time += results["Solver"][0]["Wallclock time"]
            should_repeat = iteration_cb()
            if not should_repeat:
                break
        log = streams[0].getvalue()
        node_count = self._extract_node_count(log)
        ws_value = self._extract_warm_start_value(log)
        self._termination_condition = results["Solver"][0]["Termination condition"]
        lb, ub = None, None
        if not self.is_infeasible():
            lb = results["Problem"][0]["Lower bound"]
            ub = results["Problem"][0]["Upper bound"]
        stats: MIPSolveStats = {
            "Lower bound": lb,
            "Upper bound": ub,
            "Wallclock time": total_wallclock_time,
            "Sense": self._obj_sense,
            "MIP log": log,
            "Nodes": node_count,
            "Warm start value": ws_value,
        }
        return stats

    def get_solution(self) -> Optional[Solution]:
        assert self.model is not None
        if self.is_infeasible():
            return None
        solution: Solution = {}
        for var in self.model.component_objects(Var):
            solution[str(var)] = {}
            for index in var:
                if var[index].fixed:
                    continue
                solution[str(var)][index] = var[index].value
        return solution

    def set_warm_start(self, solution: Solution) -> None:
        self._clear_warm_start()
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

    def set_instance(
        self,
        instance: Instance,
        model: Any = None,
    ) -> None:
        if model is None:
            model = instance.to_model()
        assert isinstance(model, pe.ConcreteModel)
        self.instance = instance
        self.model = model
        self._pyomo_solver.set_instance(model)
        self._update_obj()
        self._update_vars()
        self._update_constrs()

    def get_value(self, var_name: str, index: VarIndex) -> Optional[float]:
        if self.is_infeasible():
            return None
        else:
            var = self._varname_to_var[var_name]
            return var[index].value

    def get_empty_solution(self) -> Solution:
        assert self.model is not None
        solution: Solution = {}
        for var in self.model.component_objects(Var):
            svar = str(var)
            solution[svar] = {}
            for index in var:
                if var[index].fixed:
                    continue
                solution[svar][index] = None
        return solution

    def _clear_warm_start(self) -> None:
        for var in self._all_vars:
            if not var.fixed:
                var.value = None
        self._is_warm_start_available = False

    def _update_obj(self) -> None:
        self._obj_sense = "max"
        if self._pyomo_solver._objective.sense == pyomo.core.kernel.objective.minimize:
            self._obj_sense = "min"

    def _update_vars(self) -> None:
        assert self.model is not None
        self._all_vars = []
        self._bin_vars = []
        self._varname_to_var = {}
        for var in self.model.component_objects(Var):
            self._varname_to_var[var.name] = var
            for idx in var:
                self._all_vars += [var[idx]]
                if var[idx].domain == pyomo.core.base.set_types.Binary:
                    self._bin_vars += [var[idx]]

    def _update_constrs(self) -> None:
        assert self.model is not None
        self._cname_to_constr = {}
        for constr in self.model.component_objects(Constraint):
            if isinstance(constr, pe.ConstraintList):
                for idx in constr:
                    self._cname_to_constr[f"{constr.name}[{idx}]"] = constr[idx]
            else:
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

    @staticmethod
    def __extract(
        log: str,
        regexp: Optional[str],
        default: Optional[str] = None,
    ) -> Optional[str]:
        if regexp is None:
            return default
        value = default
        for line in log.splitlines():
            matches = re.findall(regexp, line)
            if len(matches) == 0:
                continue
            value = matches[0]
        return value

    def _extract_warm_start_value(self, log: str) -> Optional[float]:
        value = self.__extract(log, self._get_warm_start_regexp())
        if value is None:
            return None
        return float(value)

    def _extract_node_count(self, log: str) -> Optional[int]:
        value = self.__extract(log, self._get_node_count_regexp())
        if value is None:
            return None
        return int(value)

    def get_constraint_ids(self):
        return list(self._cname_to_constr.keys())

    def _get_warm_start_regexp(self) -> Optional[str]:
        return None

    def _get_node_count_regexp(self) -> Optional[str]:
        return None

    def relax(self) -> None:
        for var in self._bin_vars:
            lb, ub = var.bounds
            var.setlb(lb)
            var.setub(ub)
            var.domain = pyomo.core.base.set_types.Reals
            self._pyomo_solver.update_var(var)

    def get_inequality_slacks(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for (cname, cobj) in self._cname_to_constr.items():
            if cobj.equality:
                continue
            result[cname] = cobj.slack()
        return result

    def get_constraint_sense(self, cid: str) -> str:
        cobj = self._cname_to_constr[cid]
        has_ub = cobj.has_ub()
        has_lb = cobj.has_lb()
        assert (
            (not has_lb) or (not has_ub) or cobj.upper() == cobj.lower()
        ), "range constraints not supported"
        if has_lb:
            return ">"
        elif has_ub:
            return "<"
        else:
            return "="

    def get_constraint_rhs(self, cid: str) -> float:
        cobj = self._cname_to_constr[cid]
        if cobj.has_ub:
            return cobj.upper()
        else:
            return cobj.lower()

    def get_constraint_lhs(self, cid: str) -> Dict[str, float]:
        return {}

    def set_constraint_sense(self, cid: str, sense: str) -> None:
        raise NotImplementedError()

    def extract_constraint(self, cid: str) -> Constraint:
        raise NotImplementedError()

    def is_constraint_satisfied(self, cobj: Constraint, tol: float = 1e-6) -> bool:
        raise NotImplementedError()

    def is_infeasible(self) -> bool:
        return self._termination_condition == TerminationCondition.infeasible

    def get_dual(self, cid):
        raise NotImplementedError()

    def get_sense(self) -> str:
        return self._obj_sense
