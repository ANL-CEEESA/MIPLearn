#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import sys
from typing import Any, List

logger = logging.getLogger(__name__)


class RedirectOutput:
    def __init__(self, streams: List[Any]):
        self.streams = streams

    def write(self, data: Any) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, _type, _value, _traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
