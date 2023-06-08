from abc import ABC, abstractmethod

import numpy as np

from miplearn.h5 import H5File


class FeaturesExtractor(ABC):
    @abstractmethod
    def get_instance_features(self, h5: H5File) -> np.ndarray:
        pass

    @abstractmethod
    def get_var_features(self, h5: H5File) -> np.ndarray:
        pass

    @abstractmethod
    def get_constr_features(self, h5: H5File) -> np.ndarray:
        pass
