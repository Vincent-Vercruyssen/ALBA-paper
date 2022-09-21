""" Data object containing the domains. """

import numpy as np
from collections import OrderedDict


class Data:

    # TODO: implement copy functionality
    # TODO: additional functionality as needed

    def __init__(self, K):

        self.K_ = int(K)

        # data-structures
        self.domains_ = OrderedDict({})
        self.labels_ = OrderedDict({})
        self.keys_ = np.arange(0, self.K_, 1)
        self.processed_label_count_ = (
            np.ones(self.K_, dtype=int) * -1
        )  # labels used in classifier
        self.n_ = None
        self.m_ = None
        self.N_ = None
        self.last_labeled_ = None

        # transfer-related data structures
        self.transformed_domains_ = OrderedDict({})  # [to_key][from_key]
        self.transfer_weights_ = OrderedDict({})  # [to_key][from_key]
        self.relationship_matrix_ = np.zeros((self.K_, self.K_), dtype=float)  # R
        self.proximity_matrix_ = np.zeros((self.K_, self.K_), dtype=float)  # delta
        self.mmd_matrix_ = OrderedDict({})  # alpha matrix

    # SETTERS
    def set_domains_and_labels(self, domains, labels=None, keep_keys=True, copy=True):
        for new_key, key in enumerate(domains.keys()):
            if keep_keys:
                new_key = key
            self.domains_[new_key] = self.get_copy(domains[key], copy=copy)
            if labels is not None:
                self.labels_[new_key] = self.get_copy(labels[key], copy=copy)
            else:
                n = domains[key].shape[0]
                self.labels_[new_key] = np.zeros(n, dtype=int)
            # transformed domains and weights
            self.transformed_domains_[new_key] = {}
            self.transfer_weights_[new_key] = {}
            self.mmd_matrix_[new_key] = {}

        # set the number of instances in each domain + total number of instances
        self.n_ = np.array([self.get_domain_shape(key)[0] for key in self.keys_])
        self.N_ = np.sum(self.n_)
        self.m_ = self.get_domain_shape(key)[1]

    def reset_domain(self, key, X):
        self.domains_[int(key)] = np.copy(X)

    def set_new_label(self, key, index, label):
        self.labels_[int(key)][int(index)] = int(label)

    def set_transformed_domains(self, from_key, to_key, transformed):
        # automatically makes a copy
        self.transformed_domains_[to_key][from_key] = np.copy(transformed)

    def set_transfer_weights(self, from_key, to_key, weights):
        # automatically makes a copy
        self.transfer_weights_[to_key][from_key] = np.copy(weights)

    def set_processed_label_count(self, key, new_count):
        self.processed_label_count_[key] = new_count

    def set_last_labeled(self, key, index):
        self.last_labeled_ = (key, index)

    # GETTERS
    def get_last_labeled(self):
        return self.last_labeled_

    def get_domain_shape(self, key):
        return self.domains_[key].shape

    def get_domain_label_count(self, key):
        return np.count_nonzero(self.labels_[key])

    def get_domain_unlabeled_count(self, key):
        return self.domains_[key].shape[0] - np.count_nonzero(self.labels_[key])

    def get_processed_label_count(self, key):
        return self.processed_label_count_[key]

    def get_domain(self, key, copy=True):
        return self.get_copy(self.domains_[key], copy=copy)

    def get_domain_labels(self, key, copy=True):
        return self.get_copy(self.labels_[key], copy=copy)

    def get_domain_and_labels(self, key, copy=True):
        return self.get_domain(key, copy), self.get_domain_labels(key, copy)

    def get_instance_label(self, key, index):
        return self.labels_[int(key)][int(index)]

    def get_all_domains(self):
        # np.vstack copies the data
        stacked = None
        for _, X in self.domains_.items():  # respects OrderedDict
            if stacked is None:
                stacked = X
            else:
                stacked = np.vstack((stacked, X))
        return stacked

    def get_all_domains_labels(self):
        # np.concatenate copies the data
        labels = np.array([])
        for _, y in self.labels_.items():  # respects OrderedDict
            labels = np.concatenate((labels, y))
        return labels

    # TRANSFER GETTERS
    def get_transformed_domain(self, from_key, to_key, copy=True):
        """ get domain [from_key] after it is transformed to match domain [to_key] """
        return self.get_copy(self.transformed_domains_[to_key][from_key], copy=copy)

    def get_transfer_weights(self, from_key, to_key, copy=True):
        """ get domain [from_key] weights after transfer to match domain [to_key] """
        return self.get_copy(self.transfer_weights_[to_key][from_key], copy=copy)

    def get_domain_after_transfer(self, key, pos_weights=False):
        """ get domain after all transfer operations from other domains have been applied """
        # np.vstack copies the data
        stacked = None
        wixs = None
        for d_key in self.keys_:
            X = self.transformed_domains_[key][d_key]
            # transfer weights
            if pos_weights and len(self.transfer_weights_[key]) > 0:
                w = self.transfer_weights_[key][d_key]
                wixs = np.where(w > 0.0)[0]
            else:
                wixs = np.arange(0, len(X), 1)
            if stacked is None:
                stacked = X[wixs, :]
            else:
                stacked = np.vstack((stacked, X[wixs, :]))
        return stacked

    def get_domain_labels_after_transfer(self, key, pos_weights=True):
        """ get domain labels after all transfer operations from other domains have been applied """
        # np.concatenate copies the data
        labels = np.array([])
        for d_key in self.keys_:
            y = self.labels_[d_key]
            if pos_weights and len(self.transfer_weights_[key]) > 0:
                w = self.transfer_weights_[key][d_key]
                wixs = np.where(w > 0.0)[0]
            else:
                wixs = np.arange(0, len(y), 1)
            labels = np.concatenate((labels, y[wixs]))
        return labels

    def get_domain_weights_after_transfer(self, key, pos_weights=True):
        """ get domain weights after all transfer operations from other domains have been applied """
        # np.concatenate copies the data
        weights = np.array([])
        for d_key in self.keys_:
            w = self.transfer_weights_[key][d_key]
            if pos_weights:
                wixs = np.where(w > 0.0)[0]
            else:
                wixs = np.arange(0, len(w), 1)
            weights = np.concatenate((weights, w[wixs]))
        return weights

    # INTERNAL FUNCTIONS
    def get_diff_keys(self, key):
        return np.setdiff1d(self.keys_, key)

    def get_copy(self, o, copy=True):
        if copy:
            return np.copy(o)
        return o
