#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import sys
from typing import Any, List, TextIO, cast, TypeVar, Optional, Sized

logger = logging.getLogger(__name__)


class _RedirectOutput:
    def __init__(self, streams: List[Any]) -> None:
        self.streams = streams

    def write(self, data: Any) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def __enter__(self) -> Any:
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = cast(TextIO, self)
        sys.stderr = cast(TextIO, self)
        return self

    def __exit__(
        self,
        _type: Any,
        _value: Any,
        _traceback: Any,
    ) -> None:
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


T = TypeVar("T", bound=Sized)


def _none_if_empty(obj: T) -> Optional[T]:
    if len(obj) == 0:
        return None
    else:
        return obj
