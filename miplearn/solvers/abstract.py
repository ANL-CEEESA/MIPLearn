#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Hashable, List, Any

import numpy as np

from miplearn.h5 import H5File


class AbstractModel(ABC):
    _supports_basis_status = False
    _supports_sensitivity_analysis = False
    _supports_node_count = False
    _supports_solution_pool = False

    WHERE_DEFAULT = "default"
    WHERE_CUTS = "cuts"
    WHERE_LAZY = "lazy"

    def __init__(self) -> None:
        self._lazy_enforce: Optional[Callable] = None
        self._lazy_separate: Optional[Callable] = None
        self._lazy: Optional[List[Any]] = None
        self._cuts_enforce: Optional[Callable] = None
        self._cuts_separate: Optional[Callable] = None
        self._cuts: Optional[List[Any]] = None
        self._cuts_aot: Optional[List[Any]] = None
        self._where = self.WHERE_DEFAULT

    @abstractmethod
    def add_constrs(
        self,
        var_names: np.ndarray,
        constrs_lhs: np.ndarray,
        constrs_sense: np.ndarray,
        constrs_rhs: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> None:
        pass

    @abstractmethod
    def extract_after_load(self, h5: H5File) -> None:
        pass

    @abstractmethod
    def extract_after_lp(self, h5: H5File) -> None:
        pass

    @abstractmethod
    def extract_after_mip(self, h5: H5File) -> None:
        pass

    @abstractmethod
    def fix_variables(
        self,
        var_names: np.ndarray,
        var_values: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> None:
        pass

    @abstractmethod
    def optimize(self) -> None:
        pass

    @abstractmethod
    def relax(self) -> "AbstractModel":
        pass

    @abstractmethod
    def set_warm_starts(
        self,
        var_names: np.ndarray,
        var_values: np.ndarray,
        stats: Optional[Dict] = None,
    ) -> None:
        pass

    @abstractmethod
    def write(self, filename: str) -> None:
        pass

    def set_cuts(self, cuts: List) -> None:
        self.cuts_aot_ = cuts

    def lazy_enforce(self, violations: List[Any]) -> None:
        if self._lazy_enforce is not None:
            self._lazy_enforce(self, violations)

    def _lazy_enforce_collected(self) -> None:
        """Adds all lazy constraints identified in the callback as actual model constraints. Useful for generating
        a final MPS file with the constraints that were required in this run."""
        if self._lazy_enforce is not None:
            self._lazy_enforce(self, self._lazy)
