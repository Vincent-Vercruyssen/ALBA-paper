""" Transfer learning functionality.

Comments:
- Use LoCIT that transfers without considering the labels.
"""

import warnings
import numpy as np

from collections import OrderedDict, Counter
from transfertools.models import CORAL
from transfertools.models import LocIT

import methods.config as cfg
from .classifier import Classifier
from .ReliableMSTL.KernelUtil import kernel_mean_matching, mmd



def get_transfer_classifier(transfer_function, modus, transfer_settings={}):
    if len(transfer_settings) == 0:
        transfer_settings = initialize_transfer_settings(transfer_function)
    if transfer_function == 'locit':
        return LocitTransferClassifier(transfer_settings, modus)
    elif transfer_function == 'coral':
        return CoralTransferClassifier(transfer_settings, modus)
    elif transfer_function == 'pwmstl':
        return PwmstlTransferClassifier(transfer_settings, modus)
    elif transfer_function == 'none':
        return NoTransferClassifier(modus)
    else:
        raise ValueError('{} is not in [locit, coral, pwmslt, none]'.format(transfer_function))


def initialize_transfer_settings(transfer_function):
    settings = {'locit': cfg.locit_settings,
        'coral': cfg.coral_settings,
        'pwmstl': cfg.pwmstl_settings,
        'none': None}
    return settings[transfer_function.lower()]



class NoTransferClassifier:

    def __init__(self, modus='anomaly'):
        super().__init__()

        self.classifiers = OrderedDict({})
        self.modus = modus

    def apply_transfer(self, data=None):
        pass

    def fit_all(self, data, ignore_unchanged=False):
        for key in data.keys_:
            y = data.get_domain_labels(key)
            nl = np.count_nonzero(y)
            if not(ignore_unchanged) or (nl > data.get_processed_label_count(key)):
                X = data.get_domain(key)
                clf = Classifier(modus=self.modus)
                clf.fit(X, y)
                self.classifiers[key] = clf
            data.set_processed_label_count(key, nl)

    def predict(self, train_key, X, probabilities=False):
        if probabilities:
            return self.classifiers[train_key].predict_proba(X)
        else:
            return self.classifiers[train_key].predict(X)



# TODO: add bookkeeping overhead to only retrain when necessary
class LocitTransferClassifier:

    def __init__(self, transfer_settings=cfg.locit_settings, modus='anomaly'):
        super().__init__()

        self.transfer_settings = transfer_settings
        self.classifiers = OrderedDict({})
        self.modus = modus
        self.transfer_done_ = False

    def apply_transfer(self, data):
        for key in data.keys_:
            Xt = data.get_domain(key, copy=True)
            for d_key in data.get_diff_keys(key):
                Xs = data.get_domain(d_key, copy=True)
                w = np.zeros(len(Xs), dtype=float)
                transfor = LocIT(**self.transfer_settings)
                _, _, ixs = transfor.fit_transfer(Xs=Xs, Xt=Xt, return_indices=True)
                w[ixs] = 1.0
                # LoCIT does not transform, only reweights
                data.set_transformed_domains(d_key, key, Xs)
                data.set_transfer_weights(d_key, key, w)
            # store the target weights
            data.set_transformed_domains(key, key, Xt)
            data.set_transfer_weights(key, key, np.ones(len(Xt), dtype=float))
        self.transfer_done_ = True

    def fit_all(self, data, ignore_unchanged=False):
        if not(self.transfer_done_):
            self.apply_transfer(data)
        for key in data.keys_:
            y = data.get_domain_labels_after_transfer(key, pos_weights=True)
            nl = np.count_nonzero(y)
            if not(ignore_unchanged) or (nl > data.get_processed_label_count(key)):
                X = data.get_domain_after_transfer(key, pos_weights=True)
                clf = Classifier(modus=self.modus)
                clf.fit(X, y)
                self.classifiers[key] = clf
            data.set_processed_label_count(key, nl)

    def predict(self, train_key, X, probabilities=False):
        if probabilities:
            return self.classifiers[train_key].predict_proba(X)
        else:
            return self.classifiers[train_key].predict(X)



# TODO: fix issue with scaling and separate test set!
class CoralTransferClassifier:

    def __init__(self, transfer_settings=cfg.coral_settings, modus='anomaly'):
        super().__init__()

        self.transfer_settings = transfer_settings
        self.classifiers = OrderedDict({})
        self.modus = modus
        self.transfer_done_ = False

    def apply_transfer(self, data):
        if data.K_ > 2:
            warnings.warn('CORAL is not optimized for >1 source domain')
            """ Actually not problematic because for each source-target combo,
                the operation applied to the target is scaling. This operation
                is always the same, independent of the source domain considered.
            """
        for key in data.keys_:
            Xt = data.get_domain(key, copy=True)
            for d_key in data.get_diff_keys(key):
                Xs = data.get_domain(d_key, copy=True)
                transfor = CORAL(**self.transfer_settings)
                Xsn, Xtn, _ = transfor.fit_transfer(Xs=Xs, Xt=Xt, return_indices=True)
                # CORAL transforms but does not reweight
                data.set_transformed_domains(d_key, key, Xsn)
                data.set_transfer_weights(d_key, key, np.ones(len(Xs), dtype=float))
            # store the targets after transformation
            data.set_transformed_domains(key, key, Xtn)
            data.set_transfer_weights(key, key, np.ones(len(Xt), dtype=float))
        self.transfer_done_ = True

    def fit_all(self, data, ignore_unchanged=False):
        if not(self.transfer_done_):
            self.apply_transfer(data)
        for key in data.keys_:
            y = data.get_domain_labels_after_transfer(key, pos_weights=True)
            nl = np.count_nonzero(y)
            if not(ignore_unchanged) or (nl > data.get_processed_label_count(key)):
                X = data.get_domain_after_transfer(key, pos_weights=True)           # automatically selects transformed data
                clf = Classifier(modus=self.modus)
                clf.fit(X, y)
                self.classifiers[key] = clf
            data.set_processed_label_count(key, nl)

    def predict(self, train_key, X, probabilities=False):
        if probabilities:
            return self.classifiers[train_key].predict_proba(X)
        else:
            return self.classifiers[train_key].predict(X)

    

# TODO: finish implementing
class PwmstlTransferClassifier:
    
    def __init__(self, transfer_settings=cfg.pwmstl_settings, modus='anomaly'):
        super().__init__()

        self.transfer_settings = transfer_settings
        self.classifiers = OrderedDict({})
        self.modus = modus
        self.transfer_done_ = False

    def apply_transfer(self, data):
        for key in data.keys_:
            # KMM alphas - MMD distance (train_source_constant_mmd_weights)

            # compute the KMM alphas
            Xt = data.get_domain(key)
            for d_key in data.get_diff_keys(key):
                Xs = data.get_domain(d_key)
                alphas = kernel_mean_matching(Xs, Xt, kernel=transfer_settings['kernel'], B=transfer_settings['max_alpha'])
                data.mmd_matrix_[d_key][key] = alphas.flatten()
            data.mmd_matrix_[key][key] = np.ones(len(Xt), dtype=float)
            # compute the MMD distance: (train_source_constant_mmd_weights)
            for d_key in data.get_diff_keys(key):
                Xs = data.get_domain(d_key)
                mmd_score = mmd(Xs, Xt, gamma=1.0, kernel=transfer_settings['kernel'], alpha='None')
                print(mmd_score)
                data.proximity_matrix_[key][d_key] = np.exp(-1 * transfer_settings['beta2'] * (mmd_score ** transfer_settings['rho']))
            # TODO: check the correct MMD score when domains are equal (equal = no discrepancy?)
            data.proximity_matrix_[d_key][key] = 0.0

    def fit_all(self, data):
        # recompute the R matrix
        # compute omega
        pass

    def predict(self, train_key, X, probabilities=False):
        # predict for a single domain
        pass
