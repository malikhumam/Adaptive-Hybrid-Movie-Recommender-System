import numpy as np


class ItemCF:
    """Item-based CF with adjusted cosine similarity + shrinkage."""

    def __init__(self, k=30, shrink=10, damp=0.5):
        self.k, self.shrink, self.damp = k, shrink, damp

    def fit(self, R):
        self.R = R
        mask = (R > 0).astype(float)
        cnt = mask.sum(axis=1)
        global_mean = R[R > 0].mean()
        self.mu = np.where(cnt > 0, R.sum(axis=1) / np.maximum(cnt, 1), global_mean)

        # adjusted cosine: subtract each USER's mean, then cosine between item columns
        C = (R - self.mu[:, None]) * mask
        dot = C.T @ C
        norms = np.sqrt((C ** 2).sum(axis=0))
        denom = np.outer(norms, norms)
        S = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)

        # shrinkage: distrust similarities built on very few co-raters
        common = mask.T @ mask
        S *= common / (common + self.shrink)
        np.fill_diagonal(S, 0)
        self.S = S
        return self

    def predict_user(self, u):
        """Predicted ratings of user u for ALL items."""
        ru = self.R[u]
        rated = np.where(ru > 0)[0]
        if len(rated) == 0:
            return np.full(self.R.shape[1], self.mu[u])
        dev = ru[rated] - self.mu[u]
        S_sub = self.S[:, rated]                       # items x rated-items
        k = min(self.k, len(rated))
        idx = np.argpartition(-S_sub, k - 1, axis=1)[:, :k]   # k nearest rated items
        sims = np.maximum(np.take_along_axis(S_sub, idx, axis=1), 0)
        num = (sims * dev[idx]).sum(axis=1)
        den = sims.sum(axis=1)
        return self.mu[u] + num / (den + self.damp)

    def predict_matrix(self):
        return np.vstack([self.predict_user(u) for u in range(self.R.shape[0])])

    def explain(self, u, i, top=1):
        """Rated items of user u most similar to item i -> [(item, sim, rating)]"""
        rated = np.where(self.R[u] > 0)[0]
        sims = self.S[i, rated]
        order = np.argsort(-sims)[:top]
        return [(int(rated[j]), float(sims[j]), float(self.R[u, rated[j]]))
                for j in order if sims[j] > 0]