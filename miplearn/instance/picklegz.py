#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import gzip
import os
import pickle
from typing import Optional, Any, List, Hashable, cast, IO

from miplearn.instance.base import logger, Instance
from miplearn.types import VarIndex


def lazy_load(func):
    def inner(self, *args):
        if self.instance is None:
            self.instance = self._load()
            self.features = self.instance.features
            self.training_data = self.instance.training_data
        return func(self, *args)

    return inner


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

    @lazy_load
    def to_model(self) -> Any:
        assert self.instance is not None
        return self.instance.to_model()

    @lazy_load
    def get_instance_features(self) -> List[float]:
        assert self.instance is not None
        return self.instance.get_instance_features()

    @lazy_load
    def get_variable_features(self, var_name: str, index: VarIndex) -> List[float]:
        assert self.instance is not None
        return self.instance.get_variable_features(var_name, index)

    @lazy_load
    def get_variable_category(
        self,
        var_name: str,
        index: VarIndex,
    ) -> Optional[Hashable]:
        assert self.instance is not None
        return self.instance.get_variable_category(var_name, index)

    @lazy_load
    def get_constraint_features(self, cid: str) -> Optional[List[float]]:
        assert self.instance is not None
        return self.instance.get_constraint_features(cid)

    @lazy_load
    def get_constraint_category(self, cid: str) -> Optional[Hashable]:
        assert self.instance is not None
        return self.instance.get_constraint_category(cid)

    @lazy_load
    def has_static_lazy_constraints(self) -> bool:
        assert self.instance is not None
        return self.instance.has_static_lazy_constraints()

    @lazy_load
    def has_dynamic_lazy_constraints(self):
        assert self.instance is not None
        return self.instance.has_dynamic_lazy_constraints()

    @lazy_load
    def is_constraint_lazy(self, cid: str) -> bool:
        assert self.instance is not None
        return self.instance.is_constraint_lazy(cid)

    @lazy_load
    def find_violated_lazy_constraints(self, model):
        assert self.instance is not None
        return self.instance.find_violated_lazy_constraints(model)

    @lazy_load
    def build_lazy_constraint(self, model, violation):
        assert self.instance is not None
        return self.instance.build_lazy_constraint(model, violation)

    @lazy_load
    def find_violated_user_cuts(self, model):
        assert self.instance is not None
        return self.instance.find_violated_user_cuts(model)

    @lazy_load
    def build_user_cut(self, model, violation):
        assert self.instance is not None
        return self.instance.build_user_cut(model, violation)

    def _load(self) -> Instance:
        obj = read_pickle_gz(self.filename)
        assert isinstance(obj, Instance)
        return obj

    def flush(self) -> None:
        write_pickle_gz(self.instance, self.filename)


def write_pickle_gz(obj: Any, filename: str) -> None:
    logger.info(f"Writing: {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with gzip.GzipFile(filename, "wb") as file:
        pickle.dump(obj, cast(IO[bytes], file))


def read_pickle_gz(filename: str) -> Any:
    logger.info(f"Reading: {filename}")
    with gzip.GzipFile(filename, "rb") as file:
        return pickle.load(cast(IO[bytes], file))


def write_pickle_gz_multiple(objs: List[Any], dirname: str) -> None:
    for (i, obj) in enumerate(objs):
        write_pickle_gz(obj, f"{dirname}/{i:05d}.pkl.gz")
