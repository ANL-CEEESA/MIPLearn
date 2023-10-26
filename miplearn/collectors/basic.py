#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import json
import os
from io import StringIO
from os.path import exists
from typing import Callable, List

from ..h5 import H5File
from ..io import _RedirectOutput, gzip, _to_h5_filename
from ..parallel import p_umap


class BasicCollector:
    def collect(
        self,
        filenames: List[str],
        build_model: Callable,
        n_jobs: int = 1,
        progress: bool = False,
    ) -> None:
        def _collect(data_filename: str) -> None:
            h5_filename = _to_h5_filename(data_filename)
            mps_filename = h5_filename.replace(".h5", ".mps")

            if exists(h5_filename):
                # Try to read optimal solution
                mip_var_values = None
                try:
                    with H5File(h5_filename, "r") as h5:
                        mip_var_values = h5.get_array("mip_var_values")
                except:
                    pass

                if mip_var_values is None:
                    print(f"Removing empty/corrupted h5 file: {h5_filename}")
                    os.remove(h5_filename)
                else:
                    return

            with H5File(h5_filename, "w") as h5:
                streams = [StringIO()]
                with _RedirectOutput(streams):
                    # Load and extract static features
                    model = build_model(data_filename)
                    model.extract_after_load(h5)

                    # Solve LP relaxation
                    relaxed = model.relax()
                    relaxed.optimize()
                    relaxed.extract_after_lp(h5)

                    # Solve MIP
                    model.optimize()
                    model.extract_after_mip(h5)

                    # Add lazy constraints to model
                    if (
                        hasattr(model, "fix_violations")
                        and model.fix_violations is not None
                    ):
                        model.fix_violations(model, model.violations_, "aot")
                        h5.put_scalar(
                            "mip_constr_violations", json.dumps(model.violations_)
                        )

                    # Save MPS file
                    model.write(mps_filename)
                    gzip(mps_filename)

                h5.put_scalar("mip_log", streams[0].getvalue())

        if n_jobs > 1:
            p_umap(
                _collect,
                filenames,
                num_cpus=n_jobs,
                desc="collect",
                smoothing=0,
                disable=not progress,
            )
        else:
            for filename in filenames:
                _collect(filename)
