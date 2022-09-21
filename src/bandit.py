import math, random
import numpy as np
from abc import abstractmethod

from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------------------
# Bandit
# -----------------------------------------------------------------------------------------


class MAB:
    """ Generic MAB class, some methods need overriding. """

    def __init__(self, K, T=100, solver="ucb", solver_param={}):

        # parameters
        self.t = -1
        self.K = int(K)
        self.T = int(T)
        self.solver = str(solver)
        self.solver_param = solver_param

        # the counts and rewards for each arm
        self.counts = np.zeros(self.K, dtype=int)
        self.rewards = {i: [] for i in range(self.K)}

        # the policy and its reward
        self.policy = np.array([])
        self.policy_payoff = np.array([])

    def decide(self, return_estimate=False):
        """ Decide the order in which to play the arms. """

        self.t += 1

        # play every arm at least once
        iz = np.where(self.counts == 0)[0]
        if len(iz) > 0:
            index = {i: 0.0 for i in range(self.K)}
            for j in iz:
                index[j] = 1.0

        # then, make decision (purely based on history)
        else:
            if self.solver == "ucb":
                index = solver_ucb(
                    self.K, self.t, self.counts, self.rewards, **self.solver_param
                )
            elif self.solver == "egreedy":
                index = solver_egreedy(
                    self.K, self.t, self.rewards, **self.solver_param
                )
            elif self.solver == "softmax":
                index = solver_softmax(self.K, self.rewards, **self.solver_param)
            elif self.solver == "eregress":
                index = solver_eregress(
                    self.K, self.t, self.rewards, **self.solver_param
                )
            elif self.solver == "ucbregress":
                index = solver_ucbregress(
                    self.K, self.t, self.counts, self.rewards, **self.solver_param
                )
            elif self.solver == "random":
                index = solver_random(self.K, **self.solver_param)
            elif self.solver == "rotting-swa":
                index = solver_rotten_swa(
                    self.K,
                    self.T,
                    self.t,
                    self.counts,
                    self.rewards,
                    **self.solver_param
                )
            else:
                raise Exception("Invalid solver")

        # decide the order to play the arms
        sorted_index = sorted(index.items(), key=lambda x: x[1], reverse=True)
        play_order = [si[0] for si in sorted_index]

        return play_order

    def play(self, i, arms):
        """ Play the chosen arm. """

        r = arms[i].pull_arm()
        self.rewards[i].append(r)
        self.policy = np.append(self.policy, i)
        self.policy_payoff = np.append(self.policy_payoff, r)
        self.counts[i] += 1

    def get_policy_payoff(self):
        """ Compute the cumulative reward for the chosen policy. """

        return self.policy_payoff.cumsum()


# -----------------------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------------------


def solver_rotten_swa(K, T, t, counts, R, **kwargs):
    """
    Based on: https://arxiv.org/abs/1702.07274
    Non-parametric case
    """

    M = math.ceil(
        kwargs["alpha"]
        * (4 ** (2 / 3))
        * (kwargs["sigma"] ** (2 / 3))
        * (K ** (-2 / 3))
        * (T ** (2 / 3))
        * (np.log(np.sqrt(2) * T) ** (1 / 3))
    )

    if t < M * K:
        # ramp-up: selection by Round-Robin (i.e., just pick one with smallest count)
        index = {i: 1.0 / (c + 1.0) for i, c in enumerate(counts)}

    else:
        # balanced selection: select the arm that maximizes average payoff over window size M
        # requires access to the reward sequence per arm
        index = {}
        for i, rs in R.items():
            mu_i = (1 / M) * np.sum(rs[-M:])
            index[i] = mu_i

    return index


def solver_random(K):
    """
    K = the number of arms (domains)
    """

    index = {i: random.random() for i in range(K)}
    return index


def solver_ucb(K, t, counts, R):
    """
    K = the number of arms (domains)
    t = current timestep
    counts = the total number of trials
    R = the sequence of past rewards
    """

    index = {}
    for i, r in R.items():
        if len(r) > 0:
            n_i = counts[i]
            mu_i = np.mean(r)
            # bound = np.sqrt((2 * np.log(n_i)) / n_i)
            bound = np.sqrt((2 * np.log(t)) / n_i)
            index[i] = mu_i + bound
        else:
            index[i] = np.inf

    return index


def solver_egreedy(K, t, R, epsilon=0.1, decay=True):
    """
    K = the number of arms (domains)
    t = current timestep
    R = the sequence of past rewards
    epsilon = the epsilon constant
    decay = decreasing epsilon over time
    """

    if decay:
        epsilon = 1.0 / np.log(t + 0.00001)

    if np.random.uniform(0.0, 1.0) > epsilon:
        # exploitation
        index = {i: np.mean(r) for i, r in R.items()}
    else:
        # exploration
        index = {i: np.random.uniform(0.0, 1.0) for i in range(K)}

    return index


def solver_softmax(K, R):
    """
    K = the number of arms (domains)
    R = the sequence of past rewards
    """

    softmax = np.zeros(K, dtype=float)
    for i, r in R.items():
        softmax[i] = np.mean(r)

    softmax = np.exp(softmax) / np.exp(softmax).sum()
    si = np.random.choice(np.arange(0, K, 1), size=1, p=softmax)[0]
    index = {i: 0.0 for i in range(K)}
    index[si] = 1.0

    return index


def solver_ucbregress(K, t, counts, R):
    """
    K = the number of arms (domains)
    t = current timestep
    counts = the number of times each arm is played
    R = the sequence of past rewards
    epsilon = the epsilon constant
    """

    index = {}
    for i, r in R.items():
        n_i = counts[i]
        bound = np.sqrt((2 * np.log(t)) / n_i)
        if len(r) < 2:
            # no fit - predict previous value
            mu_i = r[-1]
        else:
            # try fitting curve and predict the next value
            try:
                popt, _ = curve_fit(expfit, np.arange(0, len(r), 1), r, [0.1])
                mu_i = expfit(len(r) + 1, k=popt[0])
            except:
                mu_i = r[-1]  # predict previous if issues
        index[i] = mu_i + bound

    return index


def solver_eregress(K, t, R, epsilon=0.1):
    """
    K = the number of arms (domains)
    t = current timestep
    R = the sequence of past rewards
    epsilon = the epsilon constant
    """

    if np.random.uniform(0.0, 1.0) > epsilon:
        # exploitation
        index = {}
        for i, r in R.items():
            if len(r) < 2:
                # no fit - predict previous value
                index[i] = r[-1]
            else:
                # try fitting curve and predict the next value
                try:
                    popt, _ = curve_fit(expfit, np.arange(0, len(r), 1), r, [0.1])
                    index[i] = expfit(len(r) + 1, k=popt[0])
                except:
                    index[i] = r[-1]  # predict previous if issues
    else:
        # exploration
        index = {i: np.random.uniform(0.0, 1.0) for i in range(K)}

    return index


def expfit(x, k=1.0):
    return 1.0 * np.exp(-(k * x))


# -----------------------------------------------------------------------------------------
# Arms
# -----------------------------------------------------------------------------------------


class Arm:
    """ Generic arm class. """

    def __init__(self):
        self.count = 0
        self.reward = 0.0

    @abstractmethod
    def pull_arm(self):
        self.count += 1
        return self.reward

    @abstractmethod
    def update_reward(self):
        pass


class DomainArm(Arm):
    """ Model the arm, specific to MDAL. """

    def __init__(self, metric):
        super().__init__()

        self.metric = str(metric).lower()
        self.reward = None
        self.prob0 = None
        self.pred0 = None

    def update_reward(self, prob1=None, pred1=None):
        """ Update the reward for this arm. """

        # update the reward
        if not ((self.prob0 is None) and (self.pred0 is None)):
            if self.metric == "entropy":
                self.reward = self._reward_entropy(self.prob0, prob1)
            elif self.metric == "cosine":
                self.reward = self._reward_cosine(self.pred0, pred1)
            elif self.metric == "flips":
                self.reward = self._reward_flips(self.pred0, pred1)
            else:
                raise Exception("Invalid reward metric")

        # store
        self.prob0 = prob1
        self.pred0 = pred1

        return self

    # reward mechanisms
    # reward = change in metric
    def _reward_entropy(self, p0, p1):
        # probabilities to entropy (scaled)
        H0 = self._compute_entropy_sum(p0)
        H1 = self._compute_entropy_sum(p1)
        # map to range [0, 1]
        # return ((H0 - H1) + 1.0) / 2.0
        # range [-1, 1]
        return H0 - H1

    def _reward_cosine(self, p0, p1):
        # predictions
        p0 = p0.reshape(1, -1)
        p1 = p1.reshape(1, -1)
        # range [0.0 1.0]
        return 1.0 - cosine_similarity(p0, p1)[0][0]

    def _reward_flips(self, p0, p1):
        # predictions
        return (p0 != p1).sum() / len(p0)

    def _compute_entropy_sum(self, probs):
        H = 0.0
        for p in probs:
            if not (p == 0.0 or p == 1.0):
                H += -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        Z = -len(probs) * (2.0 * 0.5 * np.log(0.5))
        return H / Z

    # # REWARD mechanisms (higher = better)
    # def _entropy_reward(self, prob):
    #     H = active_learning_scores(prob)
    #     Z = - len(H) * (2 * 0.5 * np.log(0.5))
    #     return np.sum(H) / Z

    # def _cosine_reward(self, pred0, pred1):
    #     H1 = pred0.reshape(1, -1)
    #     H2 = pred1.reshape(1, -1)
    #     return 1.0 - cosine_similarity(H1, H2)[0][0]