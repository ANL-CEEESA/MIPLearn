#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
import gc
import os
from typing import Any, Optional, List, Dict, TYPE_CHECKING
import pickle

from overrides import overrides

from miplearn.features.sample import Hdf5Sample, Sample
from miplearn.instance.base import Instance

if TYPE_CHECKING:
    from miplearn.solvers.learning import InternalSolver


class FileInstance(Instance):
    def __init__(self, filename: str) -> None:
        super().__init__()
        assert os.path.exists(filename), f"File not found: {filename}"
        self.h5 = Hdf5Sample(filename)
        self.instance: Optional[Instance] = None

    # Delegation
    # -------------------------------------------------------------------------
    @overrides
    def to_model(self) -> Any:
        assert self.instance is not None
        return self.instance.to_model()

    @overrides
    def get_instance_features(self) -> List[float]:
        assert self.instance is not None
        return self.instance.get_instance_features()

    @overrides
    def get_variable_features(self) -> Dict[str, List[float]]:
        assert self.instance is not None
        return self.instance.get_variable_features()

    @overrides
    def get_variable_categories(self) -> Dict[str, str]:
        assert self.instance is not None
        return self.instance.get_variable_categories()

    @overrides
    def get_constraint_features(self) -> Dict[str, List[float]]:
        assert self.instance is not None
        return self.instance.get_constraint_features()

    @overrides
    def get_constraint_categories(self) -> Dict[str, str]:
        assert self.instance is not None
        return self.instance.get_constraint_categories()

    @overrides
    def has_static_lazy_constraints(self) -> bool:
        assert self.instance is not None
        return self.instance.has_static_lazy_constraints()

    @overrides
    def has_dynamic_lazy_constraints(self) -> bool:
        assert self.instance is not None
        return self.instance.has_dynamic_lazy_constraints()

    @overrides
    def is_constraint_lazy(self, cid: str) -> bool:
        assert self.instance is not None
        return self.instance.is_constraint_lazy(cid)

    @overrides
    def find_violated_lazy_constraints(
        self,
        solver: "InternalSolver",
        model: Any,
    ) -> List[str]:
        assert self.instance is not None
        return self.instance.find_violated_lazy_constraints(solver, model)

    @overrides
    def enforce_lazy_constraint(
        self,
        solver: "InternalSolver",
        model: Any,
        violation: str,
    ) -> None:
        assert self.instance is not None
        self.instance.enforce_lazy_constraint(solver, model, violation)

    @overrides
    def find_violated_user_cuts(self, model: Any) -> List[str]:
        assert self.instance is not None
        return self.instance.find_violated_user_cuts(model)

    @overrides
    def enforce_user_cut(
        self,
        solver: "InternalSolver",
        model: Any,
        violation: str,
    ) -> None:
        assert self.instance is not None
        self.instance.enforce_user_cut(solver, model, violation)

    # Input & Output
    # -------------------------------------------------------------------------
    @overrides
    def free(self) -> None:
        self.instance = None
        gc.collect()

    @overrides
    def load(self) -> None:
        if self.instance is not None:
            return
        self.instance = pickle.loads(self.h5.get_bytes("pickled"))
        assert isinstance(self.instance, Instance)

    @classmethod
    def save(cls, instance: Instance, filename: str) -> None:
        h5 = Hdf5Sample(filename)
        instance_pkl = pickle.dumps(instance)
        h5.put_bytes("pickled", instance_pkl)

    @overrides
    def create_sample(self) -> Sample:
        return self.h5

    @overrides
    def get_samples(self) -> List[Sample]:
        return [self.h5]
