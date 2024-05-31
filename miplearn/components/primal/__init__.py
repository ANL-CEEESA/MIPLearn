#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Tuple, List

import numpy as np

from miplearn.h5 import H5File


def _extract_var_names_values(
    h5: H5File,
    selected_var_types: List[bytes],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_var_names, bin_var_indices = _extract_var_names(h5, selected_var_types)
    var_values = h5.get_array("mip_var_values")
    assert var_values is not None
    bin_var_values = var_values[bin_var_indices].astype(int)
    return bin_var_names, bin_var_values, bin_var_indices


def _extract_var_names(
    h5: H5File,
    selected_var_types: List[bytes],
) -> Tuple[np.ndarray, np.ndarray]:
    var_types = h5.get_array("static_var_types")
    var_names = h5.get_array("static_var_names")
    assert var_types is not None
    assert var_names is not None
    bin_var_indices = np.where(np.isin(var_types, selected_var_types))[0]
    bin_var_names = var_names[bin_var_indices]
    assert len(bin_var_names.shape) == 1
    return bin_var_names, bin_var_indices


def _extract_bin_var_names_values(
    h5: H5File,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _extract_var_names_values(h5, [b"B"])


def _extract_bin_var_names(h5: H5File) -> Tuple[np.ndarray, np.ndarray]:
    return _extract_var_names(h5, [b"B"])


def _extract_int_var_names_values(
    h5: H5File,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _extract_var_names_values(h5, [b"B", b"I"])


def _extract_int_var_names(h5: H5File) -> Tuple[np.ndarray, np.ndarray]:
    return _extract_var_names(h5, [b"B", b"I"])
