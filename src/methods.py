import math, time, sys
import numpy as np
from collections import OrderedDict

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances

from .active_learning import active_learning
from .bandit import MAB, DomainArm
from .classifier import Classifier
from .transfer_learning import get_transfer_classifier


# -----------------------------------------------------------------------------------------
# Separate detector baseline
# -----------------------------------------------------------------------------------------

""" 1. has transfer
    2. has domain selection
    3. has instance selection

Description:
------------
MAB strategies for multi-domain active learning. Without transfer.

Parameters:
-----------
1. Which reward function to use for assessing the impact of labeling? --> "mabreward"
    - entropy decrease
    - label flips
    - cosine
2. How to estimate the reward for each armed-bandit? --> "mab"
    - the classic MAB strategies
3. Which level of abstraction to choose each armed bandit? --> "abstraction_level" / "abstraction_strat"
    - 0 = each domain is an armed bandit
    - >0 = divide each domain into clusters
    - smart or naive strategy (i.e., decide yourself how to divide the clusters between domains)
4. How to select a point within each armed bandit? --> "al_strategy"
    - entropy
    - random
"""


class MABMethod:
    def __init__(
        self,
        modus="anomaly",
        transfer_function="none",
        mab="none",
        mabreward="entropy",
        mab_alpha=1.0,
        mab_sigma=0.1,
        abstraction_level=0,
        abstraction_strat="naive",
        al_strategy="random",
        query_budget=100,
        verbose=False,
    ):

        # general
        self.modus = str(modus).lower()
        self.tf_function = str(transfer_function).lower()

        # MAB specific selectors
        self.mab = str(mab).lower()
        self.mabreward = str(mabreward).lower()
        self.mab_alpha = float(mab_alpha)
        self.mab_sigma = float(mab_sigma)
        self.al_strategy = str(al_strategy).lower()
        self.abstraction_level = int(abstraction_level)
        self.abstraction_strat = str(abstraction_strat).lower()

        # other
        self.iteration_ = 0
        self.classifier = None
        self.bandit = None
        self.i = None
        self.probs0 = None
        self.query_budget = int(query_budget)
        self.verbose = bool(verbose)

    def fit_query(self, train_data, return_debug_info=False):

        # first iteration: divide the domain in clusters (smart or naive strategy)
        if self.iteration_ == 0:
            self._initialize_armed_bandits(train_data)

        # fit the classifier (transfer / no transfer)
        # this is still a single classifier per domain
        if self.iteration_ == 0:
            self.classifier = get_transfer_classifier(
                self.tf_function, self.modus
            )  # could be none
            self.classifier.apply_transfer(train_data)
        self.classifier.fit_all(train_data, ignore_unchanged=True)

        # mab strategy
        play_order = self._play_multi_armed_bandit(train_data)

        # instance selection within each domain
        all_scores = active_learning(
            train_data.keys_, train_data, self.classifier, self.al_strategy
        )

        # sort queries based on the play_order
        queries = []
        for ID in play_order:
            key = self.armed_bandits[ID]["domain"]
            ixs = self.armed_bandits[ID]["indices"]
            # get the corresponding AL scores and sort
            als = all_scores[(all_scores[:, -1] == key)][ixs]
            new_queries = [
                (int(q[2]), int(q[1])) for q in als[als[:, 0].argsort()[::-1]]
            ]
            queries.extend(new_queries)

        # print(queries)

        self.iteration_ += 1
        return queries

    def predict(self, test_data, probabilities=False):
        predictions = OrderedDict({})
        for key in test_data.keys_:
            X = test_data.get_domain(key)
            predictions[key] = self.classifier.predict(key, X, probabilities)
        return predictions

    def _initialize_armed_bandits(self, train_data):

        # keep track of the armed bandits
        # structure: key = armed bandit ID, value = {domain_key: ..., indices: [...]}
        self.armed_bandits = OrderedDict({})

        if self.abstraction_level < 2:
            for ID, key in enumerate(train_data.keys_):
                n, _ = train_data.get_domain_shape(key)
                self.armed_bandits[ID] = {
                    "domain": key,
                    "indices": np.arange(n),
                }
            return

        # divide each domain in the given number of clusters
        if self.abstraction_strat == "naive":
            ID = 0
            for _, key in enumerate(train_data.keys_):
                n, _ = train_data.get_domain_shape(key)
                X = train_data.get_domain(key)

                # cluster
                clusterer = KMeans(n_clusters=self.abstraction_level)
                labels = clusterer.fit_predict(X)

                # store label indices
                for ul in np.unique(labels):
                    ixc = np.where(labels == ul)[0]
                    if len(ixc) > 0:
                        self.armed_bandits[ID] = {
                            "domain": key,
                            "indices": ixc,
                        }
                        ID += 1

        # smart division in the number of clusters: DBSCAN
        elif self.abstraction_strat == "smart":
            ID = 0
            for _, key in enumerate(train_data.keys_):
                n, _ = train_data.get_domain_shape(key)
                X = train_data.get_domain(key)

                # pairwise distances
                D = pairwise_distances(X)
                Dsorted = np.sort(D)
                eps_est = np.median(Dsorted[:, 10])

                # cluster
                clusterer = DBSCAN(eps=eps_est, min_samples=5, metric="precomputed")
                labels = clusterer.fit_predict(D)

                # store label indices
                for ul in np.unique(labels):
                    ixc = np.where(labels == ul)[0]
                    if len(ixc) > 0:
                        self.armed_bandits[ID] = {
                            "domain": key,
                            "indices": ixc,
                        }
                        ID += 1

        else:
            raise Exception("INPUT: unknown `abstraction_strat`")

    def _play_multi_armed_bandit(self, train_data):

        # TODO: code to deal with very small clusters

        # initialize everything
        if self.iteration_ == 0:
            nb = len(self.armed_bandits)

            # init bandit and arms
            self.bandit = MAB(
                nb,
                T=self.query_budget,
                solver=self.mab,
                solver_param={"alpha": self.mab_alpha, "sigma": self.mab_sigma},
            )
            self.arms = {
                ID: DomainArm(metric=self.mabreward)
                for ID, _ in self.armed_bandits.items()
            }

        # update the reward (first time is handled in the arms itself)
        all_probs = {}
        all_preds = {}
        for key in train_data.keys_:
            X = train_data.get_domain(key)
            probs = self.classifier.predict(key, X, True)
            all_probs[key] = probs[:, 1].flatten()
            all_preds[key] = self.classifier.predict(key, X, False)

        for ID, cluster in self.armed_bandits.items():
            k = cluster["domain"]
            ixs = cluster["indices"]
            self.arms[ID].update_reward(all_probs[k][ixs], all_preds[k][ixs])

        # get reward for the played arm last time
        # this can only start from the second round
        if self.iteration_ > 0:
            last_labeled = train_data.get_last_labeled()
            for ID, cluster in self.armed_bandits.items():
                if last_labeled[0] == cluster["domain"]:
                    if last_labeled[1] in cluster["indices"]:
                        self.i = ID
                        break
            self.bandit.play(self.i, self.arms)

        # decide order to play the arms (self.i = ID of the selected arm this round)
        play_order = self.bandit.decide(return_estimate=False)

        # go from play order to the specific arm to play
        # this depends on whether the examples in the domain have already been labeled
        return play_order
