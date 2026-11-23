import os, json, time, pickle, hashlib

from data_loader import load_movielens_1m_cached
from hybrid import HybridRecommender

NEEDED = ["ml-1m/ratings.dat", "ml-1m/movies.dat"]
MODEL_CACHE = "model_cache.pkl"
CODE_FILES = ["hybrid.py", "item_cf.py", "neural_cb.py"]


def read_params():
    if os.path.exists("results.json"):
        with open("results.json") as f:
            return json.load(f).get("best_params", {})
    return {}


def _signature(params):
    h = hashlib.md5()
    h.update(json.dumps(params, sort_keys=True).encode())
    for f in NEEDED:
        h.update(f"{f}:{os.path.getsize(f)}".encode())
    for f in CODE_FILES:
        if os.path.exists(f):
            with open(f, "rb") as fh:
                h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def load_or_train(force=False, trust_cache=False, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    params = read_params()

    if os.path.exists(MODEL_CACHE) and not force:
        with open(MODEL_CACHE, "rb") as f:
            cached = pickle.load(f)
        if trust_cache or cached.get("signature") == _signature(params):
            say(f"Loaded saved trained model from {MODEL_CACHE} "
                f"({time.time() - t0:.1f}s). No training needed.")
            return cached["model"], cached["titles"]
        say("Saved model is out of date (data, parameters or model code changed): retraining.")

    say("Training models from scratch (one-time, can take a few minutes)...")
    R, X, titles, user_ids, item_ids = load_movielens_1m_cached()
    say(f"  Data loaded: {R.shape[0]} users, {R.shape[1]} movies ({time.time() - t0:.0f}s)")
    if params:
        say(f"  Using tuned parameters from results.json: {params}")
    else:
        say("  No results.json found: using default parameters.")

    model = HybridRecommender(**params).fit(R, X)

    with open(MODEL_CACHE, "wb") as f:
        pickle.dump({"model": model, "titles": titles, "signature": _signature(params)}, f)
    say(f"Trained model saved to {MODEL_CACHE} ({time.time() - t0:.0f}s total). "
        f"Next start will be instant.")
    return model, titles