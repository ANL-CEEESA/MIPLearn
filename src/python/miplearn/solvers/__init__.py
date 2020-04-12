#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import sys


class RedirectOutput(object):
    def __init__(self, streams):
        self.streams = streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __enter__(self):
        pass

    def __exit__(self, _type, _value, _traceback):
        pass
