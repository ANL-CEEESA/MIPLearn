#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

from sklearn.metrics import roc_auc_score


class ClassifierEvaluator:
    def __init__(self):
        pass

    def evaluate(self, clf, x_train, y_train):
        # FIXME: use cross-validation
        proba = clf.predict_proba(x_train)
        return roc_auc_score(y_train, proba[:, 1])
