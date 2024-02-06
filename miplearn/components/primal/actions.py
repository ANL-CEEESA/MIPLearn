#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np

from miplearn.solvers.abstract import AbstractModel

logger = logging.getLogger()


class PrimalComponentAction(ABC):
    @abstractmethod
    def perform(
        self,
        model: AbstractModel,
        var_names: np.ndarray,
        var_values: np.ndarray,
        stats: Optional[Dict],
    ) -> None:
        pass


class SetWarmStart(PrimalComponentAction):
    def perform(
        self,
        model: AbstractModel,
        var_names: np.ndarray,
        var_values: np.ndarray,
        stats: Optional[Dict],
    ) -> None:
        logger.info("Setting warm starts...")
        model.set_warm_starts(var_names, var_values, stats)


class FixVariables(PrimalComponentAction):
    def perform(
        self,
        model: AbstractModel,
        var_names: np.ndarray,
        var_values: np.ndarray,
        stats: Optional[Dict],
    ) -> None:
        logger.info("Fixing variables...")
        assert len(var_values.shape) == 2
        assert var_values.shape[0] == 1
        var_values = var_values.reshape(-1)
        model.fix_variables(var_names, var_values, stats)
        if stats is not None:
            stats["Heuristic"] = True


class EnforceProximity(PrimalComponentAction):
    def __init__(self, tol: float) -> None:
        self.tol = tol

    def perform(
        self,
        model: AbstractModel,
        var_names: np.ndarray,
        var_values: np.ndarray,
        stats: Optional[Dict],
    ) -> None:
        assert len(var_values.shape) == 2
        assert var_values.shape[0] == 1
        var_values = var_values.reshape(-1)

        constr_lhs = []
        constr_vars = []
        constr_rhs = 0.0
        for i, var_name in enumerate(var_names):
            if np.isnan(var_values[i]):
                continue
            constr_lhs.append(1.0 if var_values[i] < 0.5 else -1.0)
            constr_rhs -= var_values[i]
            constr_vars.append(var_name)

        constr_rhs += len(constr_vars) * self.tol
        logger.info(
            f"Adding proximity constraint (tol={self.tol}, nz={len(constr_vars)})..."
        )

        model.add_constrs(
            np.array(constr_vars),
            np.array([constr_lhs]),
            np.array(["<"], dtype="S"),
            np.array([constr_rhs]),
        )
        if stats is not None:
            stats["Heuristic"] = True
