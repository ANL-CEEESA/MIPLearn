#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from os.path import exists
from tempfile import NamedTemporaryFile
from typing import List, Any, Union, Dict, Callable, Optional

from miplearn.h5 import H5File
from miplearn.io import _to_h5_filename
from miplearn.solvers.abstract import AbstractModel
import shutil


class LearningSolver:
    def __init__(self, components: List[Any], skip_lp: bool = False) -> None:
        self.components = components
        self.skip_lp = skip_lp

    def fit(self, data_filenames: List[str]) -> None:
        h5_filenames = [_to_h5_filename(f) for f in data_filenames]
        for comp in self.components:
            comp.fit(h5_filenames)

    def optimize(
        self,
        model: Union[str, AbstractModel],
        build_model: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        h5_filename, mode = NamedTemporaryFile().name, "w"
        if isinstance(model, str):
            assert build_model is not None
            old_h5_filename = _to_h5_filename(model)
            model = build_model(model)
            assert isinstance(model, AbstractModel)

            # If the instance has an associate H5 file, we make a temporary copy of it,
            # then work on that copy. We keep the original file unmodified
            if exists(old_h5_filename):
                shutil.copy(old_h5_filename, h5_filename)
                mode = "r+"

        stats: Dict[str, Any] = {}
        with H5File(h5_filename, mode) as h5:
            model.extract_after_load(h5)
            if not self.skip_lp:
                relaxed = model.relax()
                relaxed.optimize()
                relaxed.extract_after_lp(h5)
            for comp in self.components:
                comp.before_mip(h5_filename, model, stats)
            model.optimize()
            model.extract_after_mip(h5)

        return stats
