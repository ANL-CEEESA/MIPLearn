#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import os
import subprocess
from typing import Callable

from ..h5 import H5File


class BranchPriorityCollector:
    def __init__(
        self,
        time_limit: float = 900.0,
        print_interval: int = 1,
        node_limit: int = 500,
    ) -> None:
        self.time_limit = time_limit
        self.print_interval = print_interval
        self.node_limit = node_limit

    def collect(self, data_filename: str, _: Callable) -> None:
        basename = data_filename.replace(".pkl.gz", "")
        env = os.environ.copy()
        env["JULIA_NUM_THREADS"] = "1"
        ret = subprocess.run(
            [
                "julia",
                "--project=.",
                "-e",
                (
                    f"using CPLEX, JuMP, MIPLearn.BB; "
                    f"BB.solve!("
                    f'    optimizer_with_attributes(CPLEX.Optimizer, "CPXPARAM_Threads" => 1),'
                    f'    "{basename}",'
                    f"    print_interval={self.print_interval},"
                    f"    time_limit={self.time_limit:.2f},"
                    f"    node_limit={self.node_limit},"
                    f")"
                ),
            ],
            check=True,
            capture_output=True,
            env=env,
        )
        h5_filename = f"{basename}.h5"
        with H5File(h5_filename, "r+") as h5:
            h5.put_scalar("bb_log", ret.stdout)
