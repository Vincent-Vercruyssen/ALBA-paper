""" Active learning funtionality. """

import numpy as np


# -----------------------------------------------------------------------------------------
# Active learning with domain preselection
# -----------------------------------------------------------------------------------------


def active_learning(
    domain_keys, train_data, classifier, al_strategy="entropy", single_classifier=False
):
    """Two options:
    1. domain_key = None --> no domain preselection, active learning over all domains
    2. domain_key = int --> domain preselection, only active learning within that domain
    """
    # decide which domains to look at
    domains = train_data.keys_

    # compute active learning scores in these domains
    all_scores = []
    for key in domains:
        n = train_data.get_domain_shape(key)[0]
        if key in domain_keys:
            if al_strategy.lower() == "entropy":
                X = train_data.get_domain(key)
                if single_classifier:
                    probs = classifier.predict_proba(X)
                else:
                    probs = classifier.predict(key, X, probabilities=True)
                probs = probs[:, 1].flatten()
                scores = entropy_scores(probs) + 0.1

            elif al_strategy.lower() == "random":
                scores = np.random.uniform(low=0.0, high=1.0, size=n) + 0.1

            else:
                raise ValueError(
                    "{} should be in [entropy, random]".format(al_strategy)
                )

        else:
            scores = np.zeros(n, dtype=float)

        # combine scores with id and domain id
        scores = np.vstack((scores, np.arange(0, n, 1)))
        scores = np.vstack((scores, np.ones(n, dtype=int) * key))
        all_scores.append(scores.T)

    return np.vstack(all_scores)


# OLD function (to make older stuff work)
def active_learning_scores(probs, al_strategy="entropy", index=1):
    if al_strategy.lower() == "entropy":
        probs = probs[:, index].flatten()
        scores = entropy_scores(probs)
    elif al_strategy.lower() == "random":
        scores = np.random.uniform(low=0.0, high=1.0, size=n)
    else:
        pass
    return scores


# -----------------------------------------------------------------------------------------
# active learning scores
# -----------------------------------------------------------------------------------------


def entropy_scores(probs):
    H = np.zeros(len(probs), dtype=float)
    for i, p in enumerate(probs):
        if abs(p - 0.0) < 1e-8 or abs(1.0 - p) < 1e-8:
            H[i] = 0.0
        else:
            H[i] = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    H = np.nan_to_num(H)
    return H
