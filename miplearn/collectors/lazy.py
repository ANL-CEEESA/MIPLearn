#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from io import StringIO
from typing import Callable

import gurobipy as gp
import numpy as np
from gurobipy import GRB, LinExpr

from ..h5 import H5File
from ..io import _RedirectOutput


class LazyCollector:
    def __init__(
        self,
        min_constrs: int = 100_000,
        time_limit: float = 900,
    ) -> None:
        self.min_constrs = min_constrs
        self.time_limit = time_limit

    def collect(
        self, data_filename: str, build_model: Callable, tol: float = 1e-6
    ) -> None:
        h5_filename = f"{data_filename}.h5"
        with H5File(h5_filename, "r+") as h5:
            streams = [StringIO()]
            lazy = None
            with _RedirectOutput(streams):
                slacks = h5.get_array("mip_constr_slacks")
                assert slacks is not None

                # Check minimum problem size
                if len(slacks) < self.min_constrs:
                    print("Problem is too small. Skipping.")
                    h5.put_array("mip_constr_lazy", np.zeros(len(slacks)))
                    return

                # Load model
                print("Loading model...")
                model = build_model(data_filename)
                model.params.LazyConstraints = True
                model.params.timeLimit = self.time_limit
                gp_constrs = np.array(model.getConstrs())
                gp_vars = np.array(model.getVars())

                # Load constraints
                lhs = h5.get_sparse("static_constr_lhs")
                rhs = h5.get_array("static_constr_rhs")
                sense = h5.get_array("static_constr_sense")
                assert lhs is not None
                assert rhs is not None
                assert sense is not None
                lhs_csr = lhs.tocsr()
                lhs_csc = lhs.tocsc()
                constr_idx = np.array(range(len(rhs)))
                lazy = np.zeros(len(rhs))

                # Drop loose constraints
                selected = (slacks > 0) & ((sense == b"<") | (sense == b">"))
                loose_constrs = gp_constrs[selected]
                print(
                    f"Removing {len(loose_constrs):,d} constraints (out of {len(rhs):,d})..."
                )
                model.remove(list(loose_constrs))

                # Filter to constraints that were dropped
                lhs_csr = lhs_csr[selected, :]
                lhs_csc = lhs_csc[selected, :]
                rhs = rhs[selected]
                sense = sense[selected]
                constr_idx = constr_idx[selected]
                lazy[selected] = 1

                # Load warm start
                var_names = h5.get_array("static_var_names")
                var_values = h5.get_array("mip_var_values")
                assert var_values is not None
                assert var_names is not None
                for (var_idx, var_name) in enumerate(var_names):
                    var = model.getVarByName(var_name.decode())
                    var.start = var_values[var_idx]

                print("Solving MIP with lazy constraints callback...")

                def callback(model: gp.Model, where: int) -> None:
                    assert rhs is not None
                    assert lazy is not None
                    assert sense is not None

                    if where == GRB.Callback.MIPSOL:
                        x_val = np.array(model.cbGetSolution(model.getVars()))
                        slack = lhs_csc * x_val - rhs
                        slack[sense == b">"] *= -1
                        is_violated = slack > tol

                        for (j, rhs_j) in enumerate(rhs):
                            if is_violated[j]:
                                lazy[constr_idx[j]] = 0
                                expr = LinExpr(
                                    lhs_csr[j, :].data, gp_vars[lhs_csr[j, :].indices]
                                )
                                if sense[j] == b"<":
                                    model.cbLazy(expr <= rhs_j)
                                elif sense[j] == b">":
                                    model.cbLazy(expr >= rhs_j)
                                else:
                                    raise RuntimeError(f"Unknown sense: {sense[j]}")

                model.optimize(callback)
                print(f"Marking {lazy.sum():,.0f} constraints as lazy...")

            h5.put_array("mip_constr_lazy", lazy)
            h5.put_scalar("mip_constr_lazy_log", streams[0].getvalue())
