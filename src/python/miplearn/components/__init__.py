#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.


def classifier_evaluation_dict(tp, tn, fp, fn):
    p = tp + fn
    n = fp + tn
    d = {
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
        "Recall": tp / p,
        "Precision": tp / (tp + fp),
    }
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
