#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
import sys
import time
import warnings
import traceback

_formatwarning = warnings.formatwarning


class TimeFormatter(logging.Formatter):
    def __init__(self, start_time, log_colors):
        super().__init__()
        self.start_time = start_time
        self.log_colors = log_colors

    def format(self, record):
        if record.levelno >= logging.ERROR:
            color = self.log_colors["red"]
        elif record.levelno >= logging.WARNING:
            color = self.log_colors["yellow"]
        else:
            color = self.log_colors["green"]
        return "%s[%12.3f]%s %s" % (
            color,
            record.created - self.start_time,
            self.log_colors["reset"],
            record.getMessage(),
        )


def formatwarning_tb(*args, **kwargs):
    s = _formatwarning(*args, **kwargs)
    tb = traceback.format_stack()
    s += "".join(tb[:-1])
    return s


def setup_logger(start_time=None, force_color=False):
    if start_time is None:
        start_time = time.time()
    if sys.stdout.isatty() or force_color:
        log_colors = {
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "reset": "\033[0m",
        }
    else:
        log_colors = {
            "green": "",
            "yellow": "",
            "red": "",
            "reset": "",
        }
    handler = logging.StreamHandler()
    handler.setFormatter(TimeFormatter(start_time, log_colors))
    logging.getLogger().addHandler(handler)
    logging.getLogger("miplearn").setLevel(logging.INFO)
    warnings.formatwarning = formatwarning_tb
    logging.captureWarnings(True)
