#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import logging
from copy import deepcopy

from miplearn.classifiers import Classifier
from miplearn.classifiers.counting import CountingClassifier
from miplearn.classifiers.evaluator import ClassifierEvaluator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AdaptiveClassifier(Classifier):
    """
    A meta-classifier which dynamically selects what actual classifier to use
    based on its cross-validation score on a particular training data set.
    """

    def __init__(self,
                 candidates=None,
                 evaluator=ClassifierEvaluator()):
        """
        Initializes the meta-classifier.
        """
        if candidates is None:
            candidates = {
                "knn(100)": {
                    "classifier": KNeighborsClassifier(n_neighbors=100),
                    "min samples": 100,
                },
                "logistic": {
                    "classifier": make_pipeline(StandardScaler(),
                                                LogisticRegression()),
                    "min samples": 30,
                },
                "counting": {
                    "classifier": CountingClassifier(),
                    "min samples": 0,
                }
            }
        self.candidates = candidates
        self.evaluator = evaluator
        self.classifier = None

    def fit(self, x_train, y_train):
        best_name, best_clf, best_score = None, None, -float("inf")
        n_samples = x_train.shape[0]
        for (name, clf_dict) in self.candidates.items():
            if n_samples < clf_dict["min samples"]:
                continue
            clf = deepcopy(clf_dict["classifier"])
            clf.fit(x_train, y_train)
            score = self.evaluator.evaluate(clf, x_train, y_train)
            if score > best_score:
                best_name, best_clf, best_score = name, clf, score
        logger.debug("Best classifier: %s (score=%.3f)" % (best_name, best_score))
        self.classifier = best_clf

    def predict_proba(self, x_test):
        return self.classifier.predict_proba(x_test)
