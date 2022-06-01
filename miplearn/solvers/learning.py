#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import time
import traceback
from typing import Optional, List, Any, cast, Dict, Tuple, Callable, IO, Union

from overrides import overrides
from p_tqdm import p_map, p_umap
from tqdm.auto import tqdm

from miplearn.features.sample import Hdf5Sample, Sample
from miplearn.components.component import Component
from miplearn.components.dynamic_lazy import DynamicLazyConstraintsComponent
from miplearn.components.dynamic_user_cuts import UserCutsComponent
from miplearn.components.objective import ObjectiveValueComponent
from miplearn.components.primal import PrimalSolutionComponent
from miplearn.features.extractor import FeaturesExtractor
from miplearn.instance.base import Instance
from miplearn.solvers import _RedirectOutput
from miplearn.solvers.internal import InternalSolver
from miplearn.solvers.pyomo.gurobi import GurobiPyomoSolver
from miplearn.types import LearningSolveStats, ConstraintName
import gzip
import pickle
import miplearn
import json
from os.path import exists
from os import remove
import pyomo.environ as pe


logger = logging.getLogger(__name__)


class PyomoFindLazyCutCallbackHandler:
    def __init__(self):
        pass

    def value(self, var):
        return var.value


class PyomoEnforceLazyCutsCallbackHandler:
    def __init__(self, opt, model):
        self.model = model
        self.opt = opt
        if not hasattr(model, "miplearn_lazy_cb"):
            model.miplearn_lazy_cb = pe.ConstraintList()

    def enforce(self, expr):
        constr = self.model.miplearn_lazy_cb.add(expr=expr)
        self.opt.add_constraint(constr)


class FileInstanceWrapper(Instance):
    def __init__(
        self, data_filename: Any, build_model: Callable, mode: Optional[str] = None
    ):
        super().__init__()
        assert data_filename.endswith(".pkl.gz")
        self.filename = data_filename
        self.sample_filename = data_filename.replace(".pkl.gz", ".h5")
        self.build_model = build_model
        self.mode = mode
        self.sample = None
        self.model = None

    @overrides
    def to_model(self) -> Any:
        if self.model is None:
            self.model = miplearn.load(self.filename, self.build_model)
        return self.model

    @overrides
    def create_sample(self) -> Sample:
        return self.sample

    @overrides
    def get_samples(self) -> List[Sample]:
        return [self.sample]

    @overrides
    def free(self) -> None:
        self.sample.file.close()

    @overrides
    def load(self) -> None:
        if self.mode is None:
            self.mode = "r+" if exists(self.sample_filename) else "w"
        self.sample = Hdf5Sample(self.sample_filename, mode=self.mode)

    @overrides
    def has_dynamic_lazy_constraints(self) -> bool:
        assert hasattr(self, "model")
        return hasattr(self.model, "_miplearn_find_lazy_cuts")

    @overrides
    def find_violated_lazy_constraints(
        self,
        solver: "InternalSolver",
        model: Any,
    ) -> Dict[ConstraintName, Any]:
        if not hasattr(self.model, "_miplearn_find_lazy_cuts"):
            return {}
        cb = PyomoFindLazyCutCallbackHandler()
        violations = model._miplearn_find_lazy_cuts(cb)
        return {json.dumps(v).encode(): v for v in violations}

    @overrides
    def enforce_lazy_constraint(
        self,
        solver: "InternalSolver",
        model: Any,
        violation: Any,
    ) -> None:
        assert isinstance(solver, GurobiPyomoSolver)
        cb = PyomoEnforceLazyCutsCallbackHandler(solver._pyomo_solver, model)
        model._miplearn_enforce_lazy_cuts(cb, violation)


class MemoryInstanceWrapper(Instance):
    def __init__(self, model: Any) -> None:
        super().__init__()
        assert model is not None
        self.model = model

    @overrides
    def to_model(self) -> Any:
        return self.model

    @overrides
    def has_dynamic_lazy_constraints(self) -> bool:
        assert hasattr(self, "model")
        return hasattr(self.model, "_miplearn_find_lazy_cuts")

    @overrides
    def find_violated_lazy_constraints(
        self,
        solver: "InternalSolver",
        model: Any,
    ) -> Dict[ConstraintName, Any]:
        cb = PyomoFindLazyCutCallbackHandler()
        violations = model._miplearn_find_lazy_cuts(cb)
        return {json.dumps(v).encode(): v for v in violations}

    @overrides
    def enforce_lazy_constraint(
        self,
        solver: "InternalSolver",
        model: Any,
        violation: Any,
    ) -> None:
        assert isinstance(solver, GurobiPyomoSolver)
        cb = PyomoEnforceLazyCutsCallbackHandler(solver._pyomo_solver, model)
        model._miplearn_enforce_lazy_cuts(cb, violation)


class _GlobalVariables:
    def __init__(self) -> None:
        self.solver: Optional[LearningSolver] = None
        self.build_model: Optional[Callable] = None
        self.filenames: Optional[List[str]] = None
        self.skip = False


# Global variables used for multiprocessing. Global variables are copied by the
# operating system when the process forks. Local variables are copied through
# serialization, which is a much slower process.
_GLOBAL = [_GlobalVariables()]


def _parallel_solve(
    idx: int,
) -> Tuple[Optional[int], Optional[LearningSolveStats]]:
    solver = _GLOBAL[0].solver
    filenames = _GLOBAL[0].filenames
    build_model = _GLOBAL[0].build_model
    skip = _GLOBAL[0].skip
    assert solver is not None
    try:
        stats = solver.solve([filenames[idx]], build_model, skip=skip)
        return idx, stats[0]
    except Exception as e:
        traceback.print_exc()
        logger.exception(f"Exception while solving {filenames[idx]}. Ignoring.")
        return idx, None


class LearningSolver:
    """
    Mixed-Integer Linear Programming (MIP) solver that extracts information
    from previous runs and uses Machine Learning methods to accelerate the
    solution of new (yet unseen) instances.

    Parameters
    ----------
    components: List[Component]
        Set of components in the solver. By default, includes
        `ObjectiveValueComponent`, `PrimalSolutionComponent`,
        `DynamicLazyConstraintsComponent` and `UserCutsComponent`.
    mode: str
        If "exact", solves problem to optimality, keeping all optimality
        guarantees provided by the MIP solver. If "heuristic", uses machine
        learning more aggressively, and may return suboptimal solutions.
    solver: Callable[[], InternalSolver]
        A callable that constructs the internal solver. If None is provided,
        use GurobiPyomoSolver.
    use_lazy_cb: bool
        If true, use native solver callbacks for enforcing lazy constraints,
        instead of a simple loop. May not be supported by all solvers.
    solve_lp: bool
        If true, solve the root LP relaxation before solving the MIP. This
        option should be activated if the LP relaxation is not very
        expensive to solve and if it provides good hints for the integer
        solution.
    """

    def __init__(
        self,
        components: Optional[List[Component]] = None,
        mode: str = "exact",
        solver: Optional[InternalSolver] = None,
        use_lazy_cb: bool = False,
        solve_lp: bool = True,
        extractor: Optional[FeaturesExtractor] = None,
        extract_lhs: bool = True,
        extract_sa: bool = True,
    ) -> None:
        if solver is None:
            solver = GurobiPyomoSolver()
        if extractor is None:
            extractor = FeaturesExtractor(
                with_sa=extract_sa,
                with_lhs=extract_lhs,
            )
        assert isinstance(solver, InternalSolver)
        self.components: Dict[str, Component] = {}
        self.internal_solver: Optional[InternalSolver] = None
        self.internal_solver_prototype: InternalSolver = solver
        self.mode: str = mode
        self.solve_lp: bool = solve_lp
        self.tee = False
        self.use_lazy_cb: bool = use_lazy_cb
        self.extractor = extractor
        if components is not None:
            for comp in components:
                self._add_component(comp)
        else:
            self._add_component(ObjectiveValueComponent())
            self._add_component(PrimalSolutionComponent(mode=mode))
            self._add_component(DynamicLazyConstraintsComponent())
            self._add_component(UserCutsComponent())
        assert self.mode in ["exact", "heuristic"]

    def _solve(
        self,
        instance: Instance,
        model: Any = None,
        discard_output: bool = False,
        tee: bool = False,
    ) -> LearningSolveStats:
        """
        Solves the given instance. If trained machine-learning models are
        available, they will be used to accelerate the solution process.

        The argument `instance` may be either an Instance object or a
        filename pointing to a pickled Instance object.

        This method adds a new training sample to `instance.training_sample`.
        If a filename is provided, then the file is modified in-place. That is,
        the original file is overwritten.

        If `solver.solve_lp_first` is False, the properties lp_solution and
        lp_value will be set to dummy values.

        Parameters
        ----------
        instance: Instance
            The instance to be solved.
        model: Any
            The corresponding Pyomo model. If not provided, it will be created.
        discard_output: bool
            If True, do not write the modified instances anywhere; simply discard
            them. Useful during benchmarking.
        tee: bool
            If true, prints solver log to screen.

        Returns
        -------
        LearningSolveStats
            A dictionary of solver statistics containing at least the following
            keys: "Lower bound", "Upper bound", "Wallclock time", "Nodes",
            "Sense", "Log", "Warm start value" and "LP value".

            Additional components may generate additional keys. For example,
            ObjectiveValueComponent adds the keys "Predicted LB" and
            "Predicted UB". See the documentation of each component for more
            details.
        """

        # Generate model
        # -------------------------------------------------------
        instance.load()
        if model is None:
            with _RedirectOutput([]):
                model = instance.to_model()

        # Initialize training sample
        # -------------------------------------------------------
        sample = instance.create_sample()

        # Initialize stats
        # -------------------------------------------------------
        stats: LearningSolveStats = {}

        # Initialize internal solver
        # -------------------------------------------------------
        self.tee = tee
        self.internal_solver = self.internal_solver_prototype.clone()
        assert self.internal_solver is not None
        assert isinstance(self.internal_solver, InternalSolver)
        self.internal_solver.set_instance(instance, model)

        # Extract features (after-load)
        # -------------------------------------------------------
        logger.info("Extracting features (after-load)...")
        initial_time = time.time()
        self.extractor.extract_after_load_features(
            instance, self.internal_solver, sample
        )
        logger.info(
            "Features (after-load) extracted in %.2f seconds"
            % (time.time() - initial_time)
        )

        callback_args = (
            self,
            instance,
            model,
            stats,
            sample,
        )

        # Solve root LP relaxation
        # -------------------------------------------------------
        lp_stats = None
        if self.solve_lp:
            logger.debug("Running before_solve_lp callbacks...")
            for component in self.components.values():
                component.before_solve_lp(*callback_args)

            logger.info("Solving root LP relaxation...")
            lp_stats = self.internal_solver.solve_lp(tee=tee)
            stats.update(cast(LearningSolveStats, lp_stats.__dict__))
            assert lp_stats.lp_wallclock_time is not None
            logger.info(
                "LP relaxation solved in %.2f seconds" % lp_stats.lp_wallclock_time
            )

            logger.debug("Running after_solve_lp callbacks...")
            for component in self.components.values():
                component.after_solve_lp(*callback_args)

            # Extract features (after-lp)
            # -------------------------------------------------------
            logger.info("Extracting features (after-lp)...")
            initial_time = time.time()
            self.extractor.extract_after_lp_features(
                self.internal_solver, sample, lp_stats
            )
            logger.info(
                "Features (after-lp) extracted in %.2f seconds"
                % (time.time() - initial_time)
            )

        # Callback wrappers
        # -------------------------------------------------------
        def iteration_cb_wrapper() -> bool:
            should_repeat = False
            for comp in self.components.values():
                if comp.iteration_cb(self, instance, model):
                    should_repeat = True
            return should_repeat

        def lazy_cb_wrapper(
            cb_solver: InternalSolver,
            cb_model: Any,
        ) -> None:
            for comp in self.components.values():
                comp.lazy_cb(self, instance, model)

        def user_cut_cb_wrapper(
            cb_solver: InternalSolver,
            cb_model: Any,
        ) -> None:
            for comp in self.components.values():
                comp.user_cut_cb(self, instance, model)

        lazy_cb = None
        if self.use_lazy_cb:
            lazy_cb = lazy_cb_wrapper

        user_cut_cb = None
        if instance.has_user_cuts():
            user_cut_cb = user_cut_cb_wrapper

        # Before-solve callbacks
        # -------------------------------------------------------
        logger.debug("Running before_solve_mip callbacks...")
        for component in self.components.values():
            component.before_solve_mip(*callback_args)

        # Solve MIP
        # -------------------------------------------------------
        logger.info("Solving MIP...")
        mip_stats = self.internal_solver.solve(
            tee=tee,
            iteration_cb=iteration_cb_wrapper,
            user_cut_cb=user_cut_cb,
            lazy_cb=lazy_cb,
        )
        assert mip_stats.mip_wallclock_time is not None
        logger.info("MIP solved in %.2f seconds" % mip_stats.mip_wallclock_time)
        stats.update(cast(LearningSolveStats, mip_stats.__dict__))
        stats["Solver"] = "default"
        stats["Gap"] = self._compute_gap(
            ub=mip_stats.mip_upper_bound,
            lb=mip_stats.mip_lower_bound,
        )
        stats["Mode"] = self.mode

        # Extract features (after-mip)
        # -------------------------------------------------------
        logger.info("Extracting features (after-mip)...")
        initial_time = time.time()
        for (k, v) in mip_stats.__dict__.items():
            sample.put_scalar(k, v)
        self.extractor.extract_after_mip_features(self.internal_solver, sample)
        logger.info(
            "Features (after-mip) extracted in %.2f seconds"
            % (time.time() - initial_time)
        )

        # After-solve callbacks
        # -------------------------------------------------------
        logger.debug("Calling after_solve_mip callbacks...")
        for component in self.components.values():
            component.after_solve_mip(*callback_args)

        # Flush
        # -------------------------------------------------------
        if not discard_output:
            instance.flush()
        instance.free()

        return stats

    def solve(
        self,
        arg: Union[Any, List[str]],
        build_model: Optional[Callable] = None,
        tee: bool = False,
        progress: bool = False,
        skip: bool = False,
    ) -> Union[LearningSolveStats, List[LearningSolveStats]]:
        if isinstance(arg, list):
            assert build_model is not None
            stats = []
            for i in tqdm(arg, disable=not progress):
                instance = FileInstanceWrapper(i, build_model)
                solved = False
                if exists(instance.sample_filename):
                    try:
                        with Hdf5Sample(instance.sample_filename, mode="r") as sample:
                            if sample.get_scalar("mip_lower_bound"):
                                solved = True
                    except OSError:
                        # File exists but it is unreadable/corrupted. Delete it.
                        remove(instance.sample_filename)
                if solved and skip:
                    stats.append({})
                else:
                    s = self._solve(instance, tee=tee)

                    # Export to gzipped MPS file
                    mps_filename = instance.sample_filename.replace(".h5", ".mps")
                    instance.model.write(
                        filename=mps_filename,
                        io_options={
                            "labeler": pe.NameLabeler(),
                            "skip_objective_sense": True,
                        },
                    )
                    with open(mps_filename, "rb") as original:
                        with gzip.open(f"{mps_filename}.gz", "wb") as compressed:
                            compressed.writelines(original)
                    remove(mps_filename)

                    stats.append(s)
            return stats
        else:
            return self._solve(MemoryInstanceWrapper(arg), tee=tee)

    def fit(
        self,
        filenames: List[str],
        build_model: Callable,
        progress: bool = False,
        n_jobs: int = 1,
    ) -> None:
        instances: List[Instance] = [
            FileInstanceWrapper(f, build_model, mode="r") for f in filenames
        ]
        self._fit(instances, progress=progress, n_jobs=n_jobs)

    def parallel_solve(
        self,
        filenames: List[str],
        build_model: Optional[Callable] = None,
        n_jobs: int = 4,
        progress: bool = False,
        label: str = "solve",
        skip: bool = False,
    ) -> List[LearningSolveStats]:
        self.internal_solver = None
        self._silence_miplearn_logger()
        _GLOBAL[0].solver = self
        _GLOBAL[0].build_model = build_model
        _GLOBAL[0].filenames = filenames
        _GLOBAL[0].skip = skip
        results = p_umap(
            _parallel_solve,
            list(range(len(filenames))),
            num_cpus=n_jobs,
            disable=not progress,
            desc=label,
        )
        stats: List[LearningSolveStats] = [{} for _ in range(len(filenames))]
        for (idx, s) in results:
            if s:
                stats[idx] = s
        self._restore_miplearn_logger()
        return stats

    def _fit(
        self,
        training_instances: List[Instance],
        n_jobs: int = 1,
        progress: bool = False,
    ) -> None:
        if len(training_instances) == 0:
            logger.warning("Empty list of training instances provided. Skipping.")
            return
        Component.fit_multiple(
            list(self.components.values()),
            training_instances,
            n_jobs=n_jobs,
            progress=progress,
        )

    def _add_component(self, component: Component) -> None:
        name = component.__class__.__name__
        self.components[name] = component

    def _silence_miplearn_logger(self) -> None:
        miplearn_logger = logging.getLogger("miplearn")
        self.prev_log_level = miplearn_logger.getEffectiveLevel()
        miplearn_logger.setLevel(logging.WARNING)

    def _restore_miplearn_logger(self) -> None:
        miplearn_logger = logging.getLogger("miplearn")
        miplearn_logger.setLevel(self.prev_log_level)

    def __getstate__(self) -> Dict:
        self.internal_solver = None
        return self.__dict__

    @staticmethod
    def _compute_gap(ub: Optional[float], lb: Optional[float]) -> Optional[float]:
        if lb is None or ub is None or lb * ub < 0:
            # solver did not find a solution and/or bound
            return None
        elif abs(ub - lb) < 1e-6:
            # avoid division by zero when ub = lb = 0
            return 0.0
        else:
            # divide by max(abs(ub),abs(lb)) to ensure gap <= 1
            return (ub - lb) / max(abs(ub), abs(lb))
