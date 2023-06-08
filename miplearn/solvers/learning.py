#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from os.path import exists
from tempfile import NamedTemporaryFile
from typing import List, Any, Union

from miplearn.h5 import H5File
from miplearn.io import _to_h5_filename
from miplearn.solvers.abstract import AbstractModel


class LearningSolver:
    def __init__(self, components: List[Any], skip_lp=False):
        self.components = components
        self.skip_lp = skip_lp

    def fit(self, data_filenames):
        h5_filenames = [_to_h5_filename(f) for f in data_filenames]
        for comp in self.components:
            comp.fit(h5_filenames)

    def optimize(self, model: Union[str, AbstractModel], build_model=None):
        if isinstance(model, str):
            h5_filename = _to_h5_filename(model)
            assert build_model is not None
            model = build_model(model)
        else:
            h5_filename = NamedTemporaryFile().name
        stats = {}
        mode = "r+" if exists(h5_filename) else "w"
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
