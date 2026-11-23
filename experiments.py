import json, time
import numpy as np
from data_loader import load_movielens_1m, split_ratings
from hybrid import HybridRecommender
from metrics import rmse, mae, precision_recall_at_k, restrict_users

HIDDENS = [16, 32]
REGS = [0.02, 0.05, 0.1]
TAUS = [1, 5, 10, 20, 40, 80, 160]
BASE = dict(k=30, shrink=10, damp=0.5)


def evaluate(P_rating, P_rank, Rb, Rt):
    """RMSE/MAE use predicted ratings; Precision/Recall use the ranking scores."""
    p, r = precision_recall_at_k(P_rank, Rb, Rt, 10)
    return {"rmse": rmse(P_rating, Rt), "mae": mae(P_rating, Rt), "p10": p, "r10": r}


def main(progress=None):
    t0 = time.time()

    def report(pct, msg):
        print(f"[{pct:3d}%] {msg} ({time.time() - t0:.0f}s)")
        if progress:
            progress(pct, msg)

    report(2, "Loading MovieLens 1M")
    R_full, X, titles, user_ids, item_ids = load_movielens_1m()
    n_items = R_full.shape[1]

    # ---------- 1. Tune on a validation split ----------
    report(5, "Tuning hidden size and regularization of the neural network")
    Rb2, Rt2 = split_ratings(R_full, test_frac=0.2, seed=2)
    m = HybridRecommender(**BASE).fit(Rb2, X)
    grid, best = [], None
    for h in HIDDENS:
        for reg in REGS:
            m.reg = reg
            m.refit_neural(h)
            s = rmse(m.P_cb, Rt2)
            grid.append({"hidden": h, "reg": reg, "rmse": s})
            print(f"    hidden={h:3d} reg={reg:<5} validation RMSE = {s:.4f}")
            if best is None or s < best[0]:
                best = (s, h, reg)
    _, best_hidden, best_reg = best
    m.reg = best_reg
    m.refit_neural(best_hidden)

    report(30, "Tuning tau")
    tau_scores = [rmse(m.P_hybrid(t), Rt2) for t in TAUS]
    best_tau = TAUS[int(np.argmin(tau_scores))]
    params = dict(BASE, hidden=best_hidden, reg=best_reg, tau=best_tau)
    report(35, f"Best parameters: {params}")

    # ---------- 2. Final evaluation on a fresh split ----------
    report(38, "Final evaluation on the test split")
    Rb, Rt = split_ratings(R_full, test_frac=0.2, seed=1)
    m = HybridRecommender(**params).fit(Rb, X)
    rating = {"Item-CF": m.P_cf, "Neural Network": m.P_cb, "Hybrid": m.P_hybrid()}
    rank = {"Item-CF": m.P_cf, "Neural Network": m.P_cb, "Hybrid": m.P_hybrid_rank()}

    P_mean = np.repeat(m.cf.mu[:, None], n_items, axis=1)
    overall = {"User mean": {"rmse": rmse(P_mean, Rt), "mae": mae(P_mean, Rt),
                             "p10": None, "r10": None}}
    for name in rating:
        overall[name] = evaluate(rating[name], rank[name], Rb, Rt)

    # ---------- 3. Breakdown by user activity ----------
    report(60, "Breaking results down by user activity")
    cnt = m.n_ratings
    groups = {"Few (<=50)": cnt <= 50,
              "Medium (51-200)": (cnt > 50) & (cnt <= 200),
              "Many (>200)": cnt > 200}
    act_rmse, act_p10, act_users = {}, {}, {}
    for g, mask in groups.items():
        act_users[g] = int(mask.sum())
        act_rmse[g], act_p10[g] = {}, {}
        for name in rating:
            act_rmse[g][name] = rmse(rating[name], restrict_users(Rt, mask))
            act_p10[g][name] = precision_recall_at_k(rank[name], Rb, Rt, 10, mask=mask)[0]

    # ---------- 4. Cold-start simulation ----------
    n_keeps = [5, 10, 20, 50, 100]
    cold = {"n_keep": n_keeps, "Item-CF": [], "Neural Network": [], "Hybrid": []}
    for idx, nk in enumerate(n_keeps):
        report(65 + int(30 * idx / len(n_keeps)), f"Cold-start simulation: {nk} ratings per user")
        rng = np.random.default_rng(0)
        Rc = np.zeros_like(Rb)
        for u in range(Rb.shape[0]):
            items = np.where(Rb[u] > 0)[0]
            if len(items) > nk:
                items = rng.choice(items, nk, replace=False)
            Rc[u, items] = Rb[u, items]
        mc = HybridRecommender(**params).fit(Rc, X)
        cold["Item-CF"].append(rmse(mc.P_cf, Rt))
        cold["Neural Network"].append(rmse(mc.P_cb, Rt))
        cold["Hybrid"].append(rmse(mc.P_hybrid(), Rt))

    results = {
        "best_params": params,
        "tuning": {"grid": grid, "tau": TAUS, "tau_rmse": tau_scores},
        "overall": overall,
        "activity_rmse": act_rmse, "activity_p10": act_p10, "activity_users": act_users,
        "cold_start": cold,
    }
    report(97, "Saving results.json")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Overall ===")
    for name, v in overall.items():
        print(name, {k: (round(x, 4) if x is not None else None) for k, x in v.items()})
    report(100, "Done")
    return results


if __name__ == "__main__":
    main()
