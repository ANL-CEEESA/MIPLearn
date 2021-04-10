#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from io import StringIO

from miplearn.solvers import _RedirectOutput


def test_redirect_output() -> None:
    import sys

    original_stdout = sys.stdout
    io = StringIO()
    with _RedirectOutput([io]):
        print("Hello world")
    assert sys.stdout == original_stdout
    assert io.getvalue() == "Hello world\n"
