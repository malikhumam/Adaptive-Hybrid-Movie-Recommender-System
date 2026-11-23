import numpy as np
from item_cf import ItemCF
from neural_cb import NeuralRecommender


def zscore_rows(P, cand):
    """Standardize each user's scores over that user's candidate (unrated) items."""
    c = np.maximum(cand.sum(axis=1, keepdims=True), 1)
    mean = (P * cand).sum(axis=1, keepdims=True) / c
    var = (((P - mean) ** 2) * cand).sum(axis=1, keepdims=True) / c
    return (P - mean) / (np.sqrt(var) + 1e-9)


class HybridRecommender:
    """
    Rating prediction : r_hat = alpha(u)*Neural + (1-alpha(u))*CF
    Ranking           : same blend, but each model's scores are first standardized
                        per user (z-scores), so alpha controls how much each model
                        INFLUENCES the order, not just the scale of its raw scores.
    alpha(u) = tau / (tau + n_u)      n_u = number of ratings of user u
    """

    def __init__(self, k=30, shrink=10, damp=0.5, tau=20,
                 hidden=16, lr=0.05, lr_user=0.1, epochs=15, reg=0.02):
        self.k, self.shrink, self.damp, self.tau = k, shrink, damp, tau
        self.hidden, self.lr, self.lr_user = hidden, lr, lr_user
        self.epochs, self.reg = epochs, reg

    def fit(self, R, X):
        self.R, self.X = R, X
        self.n_ratings = (R > 0).sum(axis=1)
        self.cf = ItemCF(self.k, self.shrink, self.damp).fit(R)
        self.P_cf = self.cf.predict_matrix()
        self.refit_neural(self.hidden)
        return self

    def refit_neural(self, hidden):
        self.hidden = hidden
        self.cb = NeuralRecommender(hidden=hidden, lr=self.lr, lr_user=self.lr_user,
                                    epochs=self.epochs, reg=self.reg).fit(self.R, self.X)
        self.P_cb = self.cb.predict_matrix()

    def alpha(self, tau=None):
        tau = self.tau if tau is None else tau
        return tau / (tau + self.n_ratings)

    def P_hybrid(self, tau=None):
        """Blended predicted ratings (used for RMSE / MAE)."""
        a = self.alpha(tau)[:, None]
        return a * self.P_cb + (1 - a) * self.P_cf

    def P_hybrid_rank(self, tau=None):
        """Blended standardized scores (used for ranking / Precision@K)."""
        a = self.alpha(tau)[:, None]
        cand = self.R == 0
        return a * zscore_rows(self.P_cb, cand) + (1 - a) * zscore_rows(self.P_cf, cand)
