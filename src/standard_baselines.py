import math, time
import numpy as np
from collections import OrderedDict

from .active_learning import active_learning
from .bandit import MAB, DomainArm
from .classifier import Classifier
from .transfer_learning import get_transfer_classifier


# -----------------------------------------------------------------------------------------
# Separate detector baseline - simply divide the budget equally per domain
# -----------------------------------------------------------------------------------------

""" Strategy: simply divide the full query budget over the number of domains and query
    equally in each domain.

    As long as the threshold is not reached, the AL strategy can freely decide which domain
    to label first. The basic principle is always: the most interesting instance gets
    queried first.
"""


class BaselineD:
    def __init__(
        self,
        modus="anomaly",
        transfer_function="none",
        al_strategy="random",
        query_budget=100,
        verbose=False,
    ):

        # general
        self.modus = str(modus).lower()
        self.tf_function = str(transfer_function).lower()
        self.al_strategy = str(al_strategy).lower()

        # other
        self.iteration_ = 0
        self.classifier = None
        self.query_budget = int(query_budget)
        self.query_budget_per_domain = None
        self.verbose = bool(verbose)

    def fit_query(self, train_data, return_debug_info=False):

        # fit the classifier (transfer / no transfer)
        if self.iteration_ == 0:
            self.classifier = get_transfer_classifier(
                self.tf_function, self.modus
            )  # could be none
            self.classifier.apply_transfer(train_data)

            # query budget for each domain
            self.query_budget_per_domain = int(self.query_budget / train_data.K_)

        self.classifier.fit_all(train_data, ignore_unchanged=True)

        # check which domains can still be labeled
        query_keys = []
        for key in train_data.keys_:
            nl = train_data.get_domain_label_count(key)
            if nl < self.query_budget_per_domain:
                query_keys.append(key)

        # compute the active learning scores for the remaining domains
        # this returns 0 scores for the domains that exceed their query budget
        all_scores = active_learning(
            query_keys, train_data, self.classifier, self.al_strategy
        )

        # return query order
        np.random.shuffle(all_scores)
        all_queries = all_scores[all_scores[:, 0].argsort()[::-1]]
        queries = [(int(q[2]), int(q[1])) for q in all_queries]

        self.iteration_ += 1
        return queries

    def predict(self, test_data, probabilities=False):
        predictions = OrderedDict({})
        for key in test_data.keys_:
            X = test_data.get_domain(key)
            predictions[key] = self.classifier.predict(key, X, probabilities)
        return predictions


# -----------------------------------------------------------------------------------------
# Separate detector baseline
# -----------------------------------------------------------------------------------------

""" 1. has transfer
    2. has domain selection
    3. has instance selection
"""


class BaselineS:
    def __init__(
        self,
        modus="anomaly",
        transfer_function="none",
        al_strategy="random",
        query_budget=100,
        verbose=False,
    ):

        # general
        self.modus = str(modus).lower()
        self.tf_function = str(transfer_function).lower()
        self.al_strategy = str(al_strategy).lower()

        # other
        self.iteration_ = 0
        self.classifier = None
        self.query_budget = int(query_budget)
        self.verbose = bool(verbose)

    def fit_query(self, train_data, return_debug_info=False):

        # fit the classifier (transfer / no transfer)
        if self.iteration_ == 0:
            self.classifier = get_transfer_classifier(
                self.tf_function, self.modus
            )  # could be none
            self.classifier.apply_transfer(train_data)
        self.classifier.fit_all(train_data, ignore_unchanged=True)

        # pool all the data and do active learning
        all_scores = active_learning(
            train_data.keys_, train_data, self.classifier, self.al_strategy
        )

        # return query order
        np.random.shuffle(all_scores)
        all_queries = all_scores[all_scores[:, 0].argsort()[::-1]]
        queries = [(int(q[2]), int(q[1])) for q in all_queries]

        self.iteration_ += 1
        return queries

    def predict(self, test_data, probabilities=False):
        predictions = OrderedDict({})
        for key in test_data.keys_:
            X = test_data.get_domain(key)
            predictions[key] = self.classifier.predict(key, X, probabilities)
        return predictions


# -----------------------------------------------------------------------------------------
# Joined baseline
# -----------------------------------------------------------------------------------------

""" 1. cannot have transfer
    2. cannot have domain selection
    3. has instance selection
"""


class BaselineJ:
    # TODO: add scaling (should all domains be scaled together)

    def __init__(
        self, modus="anomaly", al_strategy="entropy", query_budget=100, verbose=False
    ):

        # general
        self.modus = str(modus).lower()
        self.al_strategy = str(al_strategy).lower()

        # other
        self.iteration_ = 0
        self.classifier = None
        self.query_budget = int(query_budget)
        self.verbose = bool(verbose)

    def fit_query(self, train_data, return_debug_info=False):

        # fit the classifier (each iteration)
        X = train_data.get_all_domains()
        y = train_data.get_all_domains_labels()
        self.classifier = Classifier(modus=self.modus)
        self.classifier.fit(X, y)

        # decide on the instance
        all_scores = active_learning(
            train_data.keys_, train_data, self.classifier, self.al_strategy, True
        )

        # return query order
        np.random.shuffle(all_scores)
        all_queries = all_scores[all_scores[:, 0].argsort()[::-1]]
        queries = [(int(q[2]), int(q[1])) for q in all_queries]

        self.iteration_ += 1
        return queries

    def predict(self, test_data, probabilities=False):
        predictions = OrderedDict({})
        for key in test_data.keys_:
            X = test_data.get_domain(key)
            if probabilities:
                predictions[key] = self.classifier.predict_proba(X)
            else:
                predictions[key] = self.classifier.predict(X)
        return predictions


# -----------------------------------------------------------------------------------------
# Unsupervised baseline
# -----------------------------------------------------------------------------------------

""" 1. cannot have transfer
    2. cannot have domain selection
    3. cannot have instance selection
"""


class BaselineU:
    def __init__(self, modus="anomaly", query_budget=100, verbose=False):

        self.iteration_ = 0
        self.classifiers = OrderedDict({})
        self.modus = str(modus).lower()
        self.query_budget = int(query_budget)
        self.verbose = bool(verbose)

    def fit_query(self, train_data, return_debug_info=False):
        """ train_data = Data object containing training data. """

        # train classifiers once
        if self.iteration_ == 0:
            for key in train_data.keys_:
                X = train_data.get_domain(key)
                clf = Classifier(modus=self.modus)
                clf.fit(X)
                self.classifiers[key] = clf

        self.iteration_ += 1
        return []

    def predict(self, test_data, probabilities=False):
        predictions = OrderedDict({})
        for key in test_data.keys_:
            X = test_data.get_domain(key)
            if probabilities:
                predictions[key] = self.classifiers[key].predict_proba(X)
            else:
                predictions[key] = self.classifiers[key].predict(X)
        return predictions
