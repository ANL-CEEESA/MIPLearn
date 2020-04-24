#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import os
import subprocess
import tempfile
from copy import deepcopy

import numpy as np
from pyomo.core import Var
from pyomo.core.base.label import TextLabeler
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

from .component import Component
from ..extractors import Extractor, VariableFeaturesExtractor

logger = logging.getLogger(__name__)


class BranchPriorityExtractor(Extractor):
    def extract(self, instances):
        result = {}
        for instance in tqdm(instances,
                             desc="Extract (branch)",
                             disable=len(instances) < 5):
            var_split = self.split_variables(instance)
            for (category, var_index_pairs) in var_split.items():
                if category not in result:
                    result[category] = []
                for (var_name, index) in var_index_pairs:
                    result[category] += [instance.branch_priorities[var_name][index]]
        for category in result:
            result[category] = np.array(result[category])
        return result


class BranchPriorityComponent(Component):
    def __init__(self,
                 node_limit=10_000,
                 regressor=KNeighborsRegressor(n_neighbors=1),
                 ):
        self.node_limit = node_limit
        self.regressors = {}
        self.regressor_prototype = regressor

    def before_solve(self, solver, instance, model):
        logger.info("Predicting branching priorities...")
        priorities = self.predict(instance)
        solver.internal_solver.set_branching_priorities(priorities)

    def after_solve(self, solver, instance, model, results):
        pass

    def fit(self, training_instances, n_jobs=1):
        for instance in tqdm(training_instances, desc="Fit (branch)"):
            if not hasattr(instance, "branch_priorities"):
                instance.branch_priorities = self.compute_priorities(instance)
        x, y = self.x(training_instances), self.y(training_instances)
        for category in x.keys():
            self.regressors[category] = deepcopy(self.regressor_prototype)
            self.regressors[category].fit(x[category], y[category])

    def x(self, instances):
        return VariableFeaturesExtractor().extract(instances)

    def y(self, instances):
        return BranchPriorityExtractor().extract(instances)

    def compute_priorities(self, instance, model=None):
        # Create LP file
        lp_file = tempfile.NamedTemporaryFile(suffix=".lp")
        if model is None:
            model = instance.to_model()
        model.write(lp_file.name)

        # Run Julia script
        src_dirname = os.path.dirname(os.path.realpath(__file__))
        julia_dirname = "%s/../../../julia" % src_dirname
        priority_file = tempfile.NamedTemporaryFile(mode="r")
        subprocess.run(["julia",
                        "--project=%s" % julia_dirname,
                        "%s/src/branching.jl" % julia_dirname,
                        lp_file.name,
                        priority_file.name,
                        str(self.node_limit)],
                       check=True)

        # Parse output
        tokens = [line.strip().split(",") for line in priority_file.readlines()]
        lp_varname_to_priority = {t[0]: int(t[1]) for t in tokens}

        # Map priorities back to Pyomo variables
        labeler = TextLabeler()
        symbol_map = list(model.solutions.symbol_map.values())[0]
        priorities = {}
        for var in model.component_objects(Var):
            priorities[var.name] = {}
            for index in var:
                category = instance.get_variable_category(var, index)
                if category is None:
                    continue
                lp_varname = symbol_map.getSymbol(var[index], labeler)
                var_priority = lp_varname_to_priority[lp_varname]
                priorities[var.name][index] = var_priority
        return priorities

    def predict(self, instance):
        priority = {}
        x_test = self.x([instance])
        var_split = Extractor.split_variables(instance)
        for category in self.regressors.keys():
            y_test = self.regressors[category].predict(x_test[category])
            for (i, (var, index)) in enumerate(var_split[category]):
                if var not in priority.keys():
                    priority[var] = {}
                priority[var][index] = y_test[i]
        return priority
