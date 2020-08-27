#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from datetime import timedelta
import logging
import time
import sys

if sys.stdout.isatty():
    log_colors = {
        "green":  '\033[92m',
        "yellow": '\033[93m',
        "red":    '\033[91m',
        "reset":  '\033[0m',
    }
else:
    log_colors = {
        "green": "",
        "yellow": "",
        "red": "",
        "reset": "",
    }

class TimeFormatter():
    def __init__(self, start_time):
        self.start_time = start_time

    def format(self, record):
        if record.levelno >= logging.ERROR:
            color = log_colors["red"]
        elif record.levelno >= logging.WARNING:
            color = log_colors["yellow"]
        else:
            color = log_colors["green"]
        return "%s[%12.3f]%s %s" % (color,
                                    record.created - self.start_time,
                                    log_colors["reset"],
                                    record.getMessage())

def setup_logger(start_time):
    handler = logging.StreamHandler()
    handler.setFormatter(TimeFormatter(start_time))
    logging.getLogger().addHandler(handler)
    logging.getLogger("miplearn").setLevel(logging.INFO)
    lg = logging.getLogger("miplearn")
