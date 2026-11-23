import numpy as np
from data_loader import FEATURES


class NeuralRecommender:
    """
    From-scratch neural network recommender (replaces content-based filtering).

    Each movie is described ONLY by its own content features (18 genres +
    6 decades). The network never sees how other users rated the movie,
    so it behaves like content-based filtering, but with a trainable
    function instead of a hand-written cosine formula.

        h_i   = ReLU(x_i . W1 + b1)                 item encoder
        r_hat = b0 + b_u + U_u . h_i                U_u = learned user vector

    Training: mini-batch gradient descent, gradients derived by hand.
    User parameters (U_u, b_u) are updated with a per-user MEAN gradient
    inside each batch, and with their own learning rate (lr_user), so every
    user vector actually learns (dividing by the whole batch size, as in the
    first version, made them learn ~4000x too slowly).
    """

    def __init__(self, hidden=16, lr=0.05, lr_user=0.1, epochs=15, reg=0.02,
                 batch_size=4096, seed=42):
        self.hidden, self.lr, self.lr_user = hidden, lr, lr_user
        self.epochs, self.reg = epochs, reg
        self.batch_size, self.seed = batch_size, seed

    def fit(self, R, X):
        self.R, self.X = R, X
        n_users, n_items = R.shape
        d, H = X.shape[1], self.hidden
        rng = np.random.default_rng(self.seed)

        u_idx, i_idx = np.nonzero(R)
        r_val = R[u_idx, i_idx]
        n = len(r_val)

        self.W1 = rng.normal(0, 0.5, (d, H))
        self.b1 = np.zeros(H)
        self.U = rng.normal(0, 0.1, (n_users, H))
        self.bu = np.zeros(n_users)
        self.b0 = float(R[R > 0].mean())
        self.history = []

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            sq = 0.0
            for start in range(0, n, self.batch_size):
                batch = perm[start:start + self.batch_size]
                ub, ib, rb = u_idx[batch], i_idx[batch], r_val[batch]
                B = len(rb)

                x_b = X[ib]
                z = x_b @ self.W1 + self.b1
                h = np.maximum(z, 0)
                pred = self.bu[ub] + self.b0 + np.sum(self.U[ub] * h, axis=1)
                err = pred - rb
                sq += float(np.sum(err ** 2))

                # ---- user parameters: mean gradient per user in this batch ----
                cnt = np.bincount(ub, minlength=n_users)
                w = (1.0 / cnt[ub])[:, None]
                dU = (err[:, None] * h + self.reg * self.U[ub]) * w
                dbu = err * w[:, 0]

                # ---- item encoder: mean gradient over the batch ----
                dz = (err[:, None] * self.U[ub]) * (z > 0)
                dW1 = x_b.T @ dz / B + self.reg * self.W1
                db1 = dz.mean(axis=0)

                np.add.at(self.U, ub, -self.lr_user * dU)
                np.add.at(self.bu, ub, -self.lr_user * dbu)
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.b0 -= self.lr * float(err.mean())

            self.history.append(float(np.sqrt(sq / n)))   # training RMSE per epoch

        self.P = self.U @ self.W1.T      # (users x features) taste profile for the UI
        return self

    def predict_matrix(self):
        Hmat = np.maximum(self.X @ self.W1 + self.b1, 0)
        return self.bu[:, None] + self.b0 + self.U @ Hmat.T

    def explain(self, u, i, top=3):
        """Features of movie i that push this user's predicted score up most
        (gradient of the score w.r.t. each input feature)."""
        z_i = self.X[i] @ self.W1 + self.b1
        active = (z_i > 0).astype(float)
        grad = (self.U[u] * active) @ self.W1.T
        present = np.where(self.X[i] > 0)[0]
        if len(present) == 0:
            return []
        order = present[np.argsort(-grad[present])][:top]
        return [FEATURES[j] for j in order if grad[j] > 0]
