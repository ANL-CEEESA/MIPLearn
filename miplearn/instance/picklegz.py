#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import gc
import gzip
import os
import pickle
from typing import Optional, Any, List, cast, IO, TYPE_CHECKING, Dict

from overrides import overrides

from miplearn.features.sample import Sample
from miplearn.instance.base import Instance

if TYPE_CHECKING:
    from miplearn.solvers.learning import InternalSolver


class PickleGzInstance(Instance):
    """
    An instance backed by a gzipped pickle file.

    The instance is only loaded to memory after an operation is called (for example,
    `to_model`).

    Parameters
    ----------
    filename: str
        Path of the gzipped pickle file that should be loaded.
    """

    # noinspection PyMissingConstructor
    def __init__(self, filename: str) -> None:
        assert os.path.exists(filename), f"File not found: {filename}"
        self.instance: Optional[Instance] = None
        self.filename: str = filename

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

    @overrides
    def load(self) -> None:
        if self.instance is None:
            obj = read_pickle_gz(self.filename)
            assert isinstance(obj, Instance)
            self.instance = obj

    @overrides
    def free(self) -> None:
        self.instance = None  # type: ignore
        gc.collect()

    @overrides
    def flush(self) -> None:
        write_pickle_gz(self.instance, self.filename)

    @overrides
    def get_samples(self) -> List[Sample]:
        assert self.instance is not None
        return self.instance.get_samples()

    @overrides
    def create_sample(self) -> Sample:
        assert self.instance is not None
        return self.instance.create_sample()


def write_pickle_gz(obj: Any, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with gzip.GzipFile(filename, "wb") as file:
        pickle.dump(obj, cast(IO[bytes], file))


def read_pickle_gz(filename: str) -> Any:
    with gzip.GzipFile(filename, "rb") as file:
        return pickle.load(cast(IO[bytes], file))


def write_pickle_gz_multiple(objs: List[Any], dirname: str) -> None:
    for (i, obj) in enumerate(objs):
        write_pickle_gz(obj, f"{dirname}/{i:05d}.pkl.gz")
