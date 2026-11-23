import numpy as np


def rmse(P, R_test):
    u, i = np.nonzero(R_test)
    pred = np.clip(P[u, i], 1, 5)
    return float(np.sqrt(np.mean((pred - R_test[u, i]) ** 2)))


def mae(P, R_test):
    u, i = np.nonzero(R_test)
    pred = np.clip(P[u, i], 1, 5)
    return float(np.mean(np.abs(pred - R_test[u, i])))


def restrict_users(R_test, mask):
    """Keep test ratings only for users where mask is True."""
    out = R_test.copy()
    out[~mask] = 0
    return out


def precision_recall_at_k(P, R_train, R_test, k=10, threshold=4, mask=None):
    """Relevant = test rating >= threshold. Candidates = items not in train."""
    users = range(P.shape[0]) if mask is None else np.where(mask)[0]
    ps, rs = [], []
    for u in users:
        relevant = np.where(R_test[u] >= threshold)[0]
        if len(relevant) == 0:
            continue
        s = P[u].astype(float).copy()
        s[R_train[u] > 0] = -np.inf
        top = np.argsort(-s)[:k]
        hits = np.isin(top, relevant).sum()
        ps.append(hits / k)
        rs.append(hits / len(relevant))
    return float(np.mean(ps)), float(np.mean(rs))