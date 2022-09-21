""" Classifier class + classification functionality. """

import numpy as np
import pandas as pd

from scipy.stats import binom
from functools import partial
from collections import OrderedDict
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from anomatools.models import SSDO


# -----------------------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------------------

class Classifier:

    # TODO: classification: possibly implement tuning of the SVC
    # TODO: anomaly detection: implement contamination factor correctly
    # TODO: anomaly detection: better hyperparameters?

    def __init__(self, modus='anomaly', contamination=0.1, tol=1e-8, verbose=False):

        self.clf = None
        self.modus_ = str(modus).lower()
        self.c_ = float(contamination)
        self.verbose_ = bool(verbose)
        self.tol_ = float(tol)

        # modus
        if self.modus_ == 'anomaly':
            self.fit = partial(self._anomaly_fit)
            self.predict = partial(self._anomaly_predict)
            self.predict_proba = partial(self._anomaly_predict_proba)
        elif self.modus_ == 'classification':
            self.fit = partial(self._classification_fit)
            self.predict = partial(self._classification_predict)
            self.predict_proba = partial(self._classification_predict_proba)
        else:
            raise ValueError('{} nor know, pick [anomaly, classification]', self._modus)

    # CLASSIFICATION
    def _classification_fit(self, X, y=None, w=None, tuning=False):
        if y is None:
            y = np.zeros(len(X), dtype=int)
        # gamma = median of the pairwise distances in the data
        # speed-up by considering only uniformly drawn subsample of the data: use 20%
        # even with 100 %, 4000 samples with 200 features, the whole thing runs in < sec
        n, _ = X.shape
        rixs1 = np.random.choice(np.arange(0, n, 1), int(n * 0.2), replace=False)
        rixs2 = np.random.choice(np.arange(0, n, 1), int(n * 0.2), replace=False)
        D = pairwise_distances(X[rixs1, :], X[rixs2, :], metric='euclidean')
        gamma = np.median(D.flatten())
        # only use the labeled data to train the SVC
        ixl = np.where(y != 0)[0]
        Xtr = X[ixl, :]
        ytr = y[ixl, :]
        # train the SVC: probability needed, random_state does not matter too much
        self.clf = SVC(C=1.0, kernel='rbf', gamma=gamma, probability=True, class_weight='balanced')
        self.clf.fit(Xtr, ytr, sample_weight=w)
        return self

    def _classification_predict_proba(self, X):
        probabilities = self.clf.predict_proba(X)
        # make sure classes in the right order!
        c = list(self.clf.classes_)
        ixc1 = c.index(-1)
        ixc2 = c.index(1)
        probabilities = probabilities[:, [ixc1, ixc2]]
        return probabilities

    def _classification_predict(self, X):
        predictions = self.clf.predict(X)
        return predictions

    # ANOMALY DETECTION
    def _anomaly_fit(self, X, y=None, w=None):
        if y is None:
            y = np.zeros(len(X), dtype=int)
        # iforest prior
        iforest = IsolationForest(n_estimators=200, contamination=self.c_, behaviour='new')
        iforest.fit(X)
        prior = iforest.decision_function(X) * -1
        prior = (prior - min(prior)) / (max(prior) - min(prior))
        # SSDO
        ssdo = SSDO(k=30, alpha=2.3, unsupervised_prior='other', contamination=self.c_)
        ssdo.fit(X, y, prior=prior)
        self.clf = {0: iforest, 1: ssdo}
        return self

    def _anomaly_predict_proba(self, X):
        # iforest prior
        prior = self.clf[0].decision_function(X) * -1
        prior = (prior - min(prior)) / (max(prior) - min(prior))
        # SSDO: probabilities [0: normal, 1: anomaly]
        probabilities = self.clf[1].predict_proba(X, method='unify', prior=prior)
        return probabilities

    def _anomaly_predict(self, X):
        # iforest prior
        prior = self.clf[0].decision_function(X) * -1
        prior = (prior - min(prior)) / (max(prior) - min(prior))
        # SSDO: [-1: normal, 1: anomaly]
        predictions = self.clf[1].predict(X, prior=prior)
        return predictions


# -----------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------

def exceed(train_scores, test_scores, prediction=np.array([]), contamination=0.1):
    
    """
    Estimate the example-wise confidence according to the model ExCeeD provided in the paper.
    First, this method estimates the outlier probability through a Bayesian approach.
    Second, it computes the example-wise confidence by simulating to draw n other examples from the population.
    Parameters
    ----------
    train_scores   : list of shape (n_train,) containing the anomaly scores of the training set (by selected model).
    test_scores    : list of shape (n_test,) containing the anomaly scores of the test set (by selected model).
    prediction     : list of shape (n_test,) assuming 1 if the example has been classified as anomaly, 0 as normal.
    contamination  : float regarding the expected proportion of anomalies in the training set. It is the contamination factor.
    Returns
    ----------
    exWise_conf    : np.array of shape (n_test,) with the example-wise confidence for all the examples in the test set.
    
    """
    
    n = len(train_scores)
    t = len(test_scores)
    n_anom = np.int(n*contamination) #expected anomalies
    
    count_instances = np.vectorize(lambda x: np.count_nonzero(train_scores <= x)) 
    n_instances = count_instances(test_scores)

    prob_func = np.vectorize(lambda x: (1+x)/(2+n)) 
    posterior_prob = prob_func(n_instances) #Outlier probability according to ExCeeD
    
    conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
    exWise_conf = conf_func(posterior_prob)
    #np.place(exWise_conf, prediction == 0, 1 - exWise_conf[prediction == 0]) # if the example is classified as normal,
                                                                             # use 1 - confidence.

    # 2D array (1st column = normal confidences, 2nd column = anomaly confidences)
    confidences = np.vstack((1.0 - exWise_conf, exWise_conf)).T

    return confidences