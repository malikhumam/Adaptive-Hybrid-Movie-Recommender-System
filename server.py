import os, io, json, zipfile, time, threading, urllib.request, webbrowser, sys
from typing import Optional

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from data_loader import FEATURES
from model_store import load_or_train

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
NEEDED = ["ml-1m/ratings.dat", "ml-1m/movies.dat"]

state = {"model": None, "titles": None}


def dataset_ready():
    return all(os.path.exists(f) for f in NEEDED)


def download_dataset():
    print("Dataset not found. Downloading MovieLens 1M (~6 MB)...")
    data = urllib.request.urlopen(URL, timeout=180).read()
    print("Unzipping...")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(".")
    print("Dataset ready.")


def load_everything():
    """Loads the saved trained model if available (fast); otherwise trains once and saves it.
    Run `python server.py --use-cache` on presentation day to skip all checks."""
    t0 = time.time()
    model, titles = load_or_train(trust_cache="--use-cache" in sys.argv)
    state["model"], state["titles"] = model, titles
    print(f"Ready in {time.time() - t0:.0f}s.")


app = FastAPI(title="Hybrid Movie Recommender")


def need_model():
    if state["model"] is None:
        raise HTTPException(503, "Model not ready")
    return state["model"]


def reason_cf(m, t, u, i):
    ex = m.cf.explain(u, i, top=1)
    if not ex:
        return "Little item-neighbour evidence for this movie"
    j, sim, r = ex[0]
    return f"Because you rated **{t[j]}** {int(r)}★ (similarity {sim:.2f})"


def reason_cb(m, u, i):
    f = m.cb.explain(u, i)
    return ("Matches your taste for **" + ", ".join(f) + "**") if f else "No strong feature match"


# ------------------------------------------------------------ API
@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/meta")
def meta():
    m = state["model"]
    return {
        "n_users": int(m.R.shape[0]),
        "n_items": int(m.R.shape[1]),
        "params": {"k": m.k, "shrink": m.shrink, "damp": m.damp, "hidden": m.hidden, "tau": m.tau},
        "results_ready": os.path.exists("results.json"),
    }


@app.get("/api/results")
def results():
    if not os.path.exists("results.json"):
        raise HTTPException(404, "No results.json found")
    with open("results.json") as f:
        return json.load(f)


@app.get("/api/pick-user")
def pick_user(kind: str = "light"):
    m = need_model()
    c = m.n_ratings
    masks = {"light": c <= 50, "medium": (c > 50) & (c <= 200), "heavy": c > 200}
    if kind not in masks:
        raise HTTPException(400, "kind must be light, medium or heavy")
    pool = np.where(masks[kind])[0]
    if len(pool) == 0:
        raise HTTPException(404, "No users in that group")
    return {"uid": int(np.random.choice(pool)) + 1}


@app.get("/api/recommend/{uid}")
def recommend(uid: int, n: int = 10, tau: Optional[float] = None):
    m, t = need_model(), state["titles"]
    if not 1 <= uid <= m.R.shape[0]:
        raise HTTPException(400, f"User id must be between 1 and {m.R.shape[0]}")
    u = uid - 1
    n = max(1, min(n, 30))
    tau = float(m.tau if tau is None else tau)
    n_r = int(m.n_ratings[u])
    alpha = tau / (tau + n_r)
    rated = m.R[u] > 0

    # Ranking uses per-user standardized scores, so alpha really controls how much
    # each model influences the ORDER. The displayed rating is the plain blend.
    cand = ~rated

    def z(x):
        return (x - x[cand].mean()) / (x[cand].std() + 1e-9)

    raw_h = alpha * m.P_cb[u] + (1 - alpha) * m.P_cf[u]
    rank_h = alpha * z(m.P_cb[u]) + (1 - alpha) * z(m.P_cf[u])
    rows = {"cf": (m.P_cf[u], m.P_cf[u]),
            "cb": (m.P_cb[u], m.P_cb[u]),
            "hybrid": (rank_h, raw_h)}
    recs = {}
    for key, (rank_row, show_row) in rows.items():
        s = rank_row.astype(float).copy()
        s[rated] = -np.inf
        top = np.argsort(-s)[:n]
        items = []
        for rank, i in enumerate(top, 1):
            i = int(i)
            if key == "cf":
                why = reason_cf(m, t, u, i)
            elif key == "cb":
                why = reason_cb(m, u, i)
            else:
                main = reason_cb(m, u, i) if alpha >= 0.5 else reason_cf(m, t, u, i)
                why = f"[CF {1 - alpha:.0%} · Neural {alpha:.0%}] " + main
            items.append({"rank": rank, "id": i + 1, "title": t[i],
                          "score": float(np.clip(show_row[i], 1, 5)), "reason": why})
        recs[key] = items

    pu = m.cb.P[u]
    idx = np.argsort(-np.abs(pu))[:8]
    idx = idx[np.argsort(pu[idx])]
    profile = [{"feature": FEATURES[j], "value": float(pu[j])} for j in idx]

    r = np.where(rated)[0]
    order = r[np.argsort(-m.R[u, r], kind="stable")][:8]
    top_rated = [{"title": t[i], "rating": int(m.R[u, i])} for i in order]

    return {"uid": uid, "n": n, "tau": tau, "n_ratings": n_r, "alpha": alpha,
            "mean": float(m.cf.mu[u]), "recs": recs, "profile": profile,
            "top_rated": top_rated}


@app.get("/api/dataset")
def dataset():
    m = need_model()
    R = m.R
    nnz = int((R > 0).sum())
    counts, edges = np.histogram(m.n_ratings, bins=15)
    fc = m.X.sum(axis=0)
    order = np.argsort(fc)
    return {
        "users": int(R.shape[0]), "items": int(R.shape[1]), "ratings": nnz,
        "sparsity": 1 - nnz / R.size,
        "rating_counts": [int((R == r).sum()) for r in range(1, 6)],
        "activity": {"labels": [f"{int(edges[k])}-{int(edges[k + 1])}" for k in range(len(counts))],
                     "counts": counts.tolist()},
        "features": [{"name": FEATURES[j], "count": int(fc[j])} for j in order],
    }


if __name__ == "__main__":
    if not dataset_ready():
        try:
            download_dataset()
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print(f"Please download it manually from {URL} and unzip into this folder as ml-1m/")
            sys.exit(1)

    load_everything()

    print("\nStarting server at http://127.0.0.1:8000 ...\n")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
