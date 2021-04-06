#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.
from typing import Dict


def classifier_evaluation_dict(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> Dict[str, float]:
    p = tp + fn
    n = fp + tn
    d: Dict = {
        "Predicted positive": fp + tp,
        "Predicted negative": fn + tn,
        "Condition positive": p,
        "Condition negative": n,
        "True positive": tp,
        "True negative": tn,
        "False positive": fp,
        "False negative": fn,
        "Accuracy": (tp + tn) / (p + n),
        "F1 score": (2 * tp) / (2 * tp + fp + fn),
    }

    if p > 0:
        d["Recall"] = tp / p
    else:
        d["Recall"] = 1.0

    if tp + fp > 0:
        d["Precision"] = tp / (tp + fp)
    else:
        d["Precision"] = 1.0

    t = (p + n) / 100.0
    d["Predicted positive (%)"] = d["Predicted positive"] / t
    d["Predicted negative (%)"] = d["Predicted negative"] / t
    d["Condition positive (%)"] = d["Condition positive"] / t
    d["Condition negative (%)"] = d["Condition negative"] / t
    d["True positive (%)"] = d["True positive"] / t
    d["True negative (%)"] = d["True negative"] / t
    d["False positive (%)"] = d["False positive"] / t
    d["False negative (%)"] = d["False negative"] / t
    return d
