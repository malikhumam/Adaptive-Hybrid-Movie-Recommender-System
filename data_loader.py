import numpy as np
import pickle
import os

CACHE_FILE = "ml1m_cache.pkl"

RATINGS_FILE = "ml-1m/ratings.dat"
MOVIES_FILE = "ml-1m/movies.dat"

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
DECADES = ["Pre-1960", "1960s", "1970s", "1980s", "1990s", "2000s"]
FEATURES = GENRES + DECADES


def _read_lines(path):
    with open(path, encoding="latin-1") as f:
        return [l.rstrip("\n") for l in f]


def load_movielens_1m(ratings_path=RATINGS_FILE, movies_path=MOVIES_FILE):
    """
    1M format differs from 100k:
      ratings.dat : UserID::MovieID::Rating::Timestamp
      movies.dat  : MovieID::Title (Year)::Genre1|Genre2|...
    Movie IDs have gaps (go up to 3952 but only 3706 exist), so we
    remap real IDs -> dense array indices 0..n_items-1.

    Returns: R (users x items), X (items x features), titles,
             user_ids, item_ids  (row/col i -> original ID)
    """
    # ---------- movies.dat ----------
    movie_rows = {}
    for line in _read_lines(movies_path):
        mid, title, genres = line.split("::")
        movie_rows[int(mid)] = (title, genres.split("|"))

    item_ids = sorted(movie_rows.keys())
    item_index = {mid: idx for idx, mid in enumerate(item_ids)}
    n_items = len(item_ids)

    titles = [""] * n_items
    G = np.zeros((n_items, len(GENRES)))
    years = np.zeros(n_items, dtype=int)
    for mid, (title, glist) in movie_rows.items():
        j = item_index[mid]
        titles[j] = title
        t = title.strip()
        if t.endswith(")") and t[-5:-1].isdigit():
            years[j] = int(t[-5:-1])
        for g in glist:
            if g in GENRES:
                G[j, GENRES.index(g)] = 1.0

    D = np.zeros((n_items, len(DECADES)))
    for j, y in enumerate(years):
        if y == 0:
            continue
        d = 0 if y < 1960 else 5 if y >= 2000 else (y - 1960) // 10 + 1
        D[j, d] = 1.0
    X = np.hstack([G, D])

    # ---------- ratings.dat ----------
    user_ids_set = set()
    raw = []
    for line in _read_lines(ratings_path):
        u, m, r, _ = line.split("::")
        u, m, r = int(u), int(m), float(r)
        user_ids_set.add(u)
        raw.append((u, m, r))

    user_ids = sorted(user_ids_set)
    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    n_users = len(user_ids)

    R = np.zeros((n_users, n_items))
    for u, m, r in raw:
        R[user_index[u], item_index[m]] = r

    return R, X, titles, user_ids, item_ids


def split_ratings(R, test_frac=0.2, seed=0, min_ratings=5):
    """
    1M has no ready-made train/test folds (unlike 100k's u1..u5),
    so we hold out test_frac of EACH user's ratings at random.
    Users with very few ratings keep everything in training.
    """
    rng = np.random.default_rng(seed)
    Rb, Rt = np.zeros_like(R), np.zeros_like(R)
    for u in range(R.shape[0]):
        items = np.where(R[u] > 0)[0]
        if len(items) < min_ratings:
            Rb[u, items] = R[u, items]
            continue
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_frac))
        test_items, train_items = items[:n_test], items[n_test:]
        Rb[u, train_items] = R[u, train_items]
        Rt[u, test_items] = R[u, test_items]
    return Rb, Rt

def load_movielens_1m_cached(ratings_path=RATINGS_FILE, movies_path=MOVIES_FILE,
                              cache_path=CACHE_FILE):
    """
    Same return value as load_movielens_1m(), but reuses a cached, already-parsed
    copy if one exists and the source .dat files haven't changed since it was made.
    First run: parses the .dat files (slow) and writes the cache.
    Later runs: loads the cache directly (fast, no re-parsing).
    """
    src_mtime = max(os.path.getmtime(ratings_path), os.path.getmtime(movies_path))

    if os.path.exists(cache_path) and os.path.getmtime(cache_path) > src_mtime:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data = load_movielens_1m(ratings_path, movies_path)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data