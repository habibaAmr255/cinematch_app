# 🎬 CineMatch: movie recommender (Streamlit)

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
First launch takes ~15 s (builds the TF-IDF baseline, cached afterwards in `models/`).

## What's inside
| Part | Where |
|---|---|
| Login / sign up / guest | `views/welcome.py`, `core/storage.py` (SQLite file `data/app.db`, PBKDF2-hashed passwords) |
| Browse + filters, Similar movies, My lists (favorites / watched / "for you") | `views/` |
| Chatbot (offline, rule-based, English only) | `chatbot/parser.py` + `chatbot/understand.py` (patterns) → `chatbot/engine.py` (routing, memory, replies), `chatbot/facts.py` (dataset facts) |
| Insights page (charts from the notebook) | `views/insights.py` |
| Recommenders | `recommender/` (`base.py` is the contract, `stacked_model.py` = the notebook's models) |
| Data | `data/movies_clean.csv.gz` (made by `scripts/prepare_data.py`) |

Guests can browse, search, chat and get recommendations. Logged-in users can also favorite
movies and mark them as watched (buttons on every card, or by chatting:
*"add the second one to my favorites"*, *"mark Inception as watched"*).

## The models from the notebook
`movies_insights_V7.ipynb` (sections "Saving everything for the app" and
"Recommendation models") builds a hybrid embedding plus two trained re-rankers.
`scripts/build_ml_models.py` re-runs those cells outside Jupyter and writes what the app
needs into `models/`:

```bash
python scripts/build_ml_models.py --data "path/to/df_sampled.csv"      # ~10-20 min
```

| File | Needed? | What it is |
|---|---|---|
| `blocks.npz` | yes | the 8 embedding blocks (text, genres, keywords, cast, directors, writers, companies, meta) |
| `ranker.joblib` | yes | the trained XGBoost and the stacked model (XGBoost + KNN + logistic regression) |
| `fitted.joblib` | optional | fusion weights and the free-text encoder spec (defaults = the notebook's) |
| `ok_mask.npy` | optional | rows that must never be recommended (duplicates, "trilogy" box sets) |
| `app_meta.pkl` | optional | only used to check the rows line up with `data/movies_clean.csv.gz` |

`recsys_utils.py` (project root) must stay where it is: `fitted.joblib` points to it.

Once the files are there, four extra entries appear in the sidebar **Recommendation model**
dropdown (and as tabs under the Similar page's "Compare all models"):

| Entry | What it does |
|---|---|
| **Hybrid (cosine)** | the fused embedding only, no trained model |
| **Hybrid + Stack (50/50)** | hybrid search, then half cosine / half stacked-model score (notebook `mode="both"`) |
| **Stacked model (XGB+KNN+LR)** | hybrid search for the best candidates, re-ranked by the stack only |
| **XGBoost re-ranker** | same, with XGBoost alone |

The dropdown drives **every** page: the Similar page, My lists → "For you" and the chatbot.
They all follow the notebook's `recommend_v2`: the filters decide what is allowed, the liked
movies (or the 5 best matches of a typed theme, or the 10 best-rated movies inside the
filters) form the profile, the hybrid embedding finds the nearest candidates, the model
re-ranks them, and the "Favor well-rated movies" slider is the weight of the weighted rating.
"Most popular" / "Newest" / no-input requests skip the model, because there is nothing to
rank by similarity.

Notes
* `models/ranker.joblib` is a pickle: build it with the same scikit-learn / xgboost versions
  the app runs (the shipped one was built with scikit-learn 1.9.1 and xgboost 3.2.0).
  When the versions differ the notebook models are skipped with a reason in the terminal,
  and the app still starts on the baseline.
* **Text block.** The notebook encodes title + tagline + overview + keywords with SBERT
  (`all-MiniLM-L6-v2`, 384 dims). `scripts/build_ml_models.py` uses the notebook's own
  TF-IDF + SVD fallback unless you pass `--sbert`; that needs
  `pip install sentence-transformers` (it pulls in torch). Everything works either way, but
  SBERT is noticeably better on free-text themes.
* Free text also works without SBERT: the hybrid models then score themes with the baseline's
  TF-IDF vectors.
* Row *i* of every file must be row *i* of `data/movies_clean.csv.gz`. The build script checks
  this at the end (`scripts/prepare_data.py` reproduces the notebook's row order: read
  `df_sampled.csv` → drop duplicate `tconst` → drop `adult`).

Any other model that gives one vector per movie (SBERT only, SVD, autoencoder…) can still be
added as `models/<name>.npy` or `models/<name>.npz` (optionally with `ids` to re-align rows)
and shows up under that name. To change which notebook models are offered, edit `SPECS` in
`recommender/stacked_model.py`. For a model that is not an embedding matrix, subclass
`BaseRecommender` (2 methods: `similarity_to`, `similarity_to_text`) and register it in
`recommender/registry.py` → `CUSTOM`.

## Chatbot

Offline and rule-based, but it is a conversation, not a query box: it describes films, answers
questions about them, remembers what you were just doing and asks when it does not understand.
Everything it says comes from the dataset — it never invents a movie or a fact.

| What | Examples |
|---|---|
| **Describe a film** | `what is Inception about` · `describe The Dark Knight` · `tell me more about Parasite` |
| **Ask a fact** | `how long is The Godfather` · `who directed Interstellar` · `who is in Pulp Fiction` · `what year was Alien made` · `what genre is Se7en` · `is Se7en good` · `how popular is Parasite` · `what is the budget of Avatar` |
| **Compare** | `compare The Godfather and GoodFellas` (side-by-side table + a verdict) |
| **Follow up** | `show me more` · `the first one` · `what is it about` · `is it good` · `why did you pick that` · `make it shorter` · `not that one` · `I have 2 hours` |
| **Find something** | `scary movies from the 90s` · `top 5 highly rated Japanese animation` · `movies directed by Christopher Nolan` · `movies with Tom Hanks` · `space heist` · `I liked Inception and Interstellar` · `surprise me` |
| **Just talk** | `hi` · `how are you` · `who are you` · `thanks` · `I'm bored` · `ok` · `yes` / `no` |

Rules it follows:
* Facts come from the data. If a field is missing it says so (e.g. there are no role names, only the
  billed cast, so `who plays the Joker in The Dark Knight` lists the cast and points at IMDb).
* When it cannot understand you it **asks one question** instead of returning a random list
  (`I want something good` → offers a mood, a genre or `surprise me`). Same when the only words you
  gave it match nothing in the dataset.
* It never repeats itself: `show me more` skips everything you have already been shown.
* Non-English input gets a "please type in English" reply.
* With filters that are too narrow it relaxes them one by one and says which it dropped.

Files: `chatbot/parser.py` (the original patterns) → `chatbot/understand.py` (descriptions, facts,
comparisons, follow-ups, small talk) → `chatbot/engine.py` (routing, replies, memory) with
`chatbot/facts.py` reading the dataset, and `chatbot/lookup.py` for titles and people.
`recommender/*.why()` supplies the "why did you pick that" reason — for the notebook models it is the
embedding blocks that contributed most, the same idea as the notebook's `why_simple`.

## Deploying

### What ships
The repo is about **180 MB**, and that is fine: every file is under GitHub's 100 MB per-file
limit, so no Git LFS is needed. The large files **are committed on purpose** — the deployed app
cannot work without them:

| Committed | Size | Why |
|---|---|---|
| `models/fitted.joblib` | 67.9 MB | fusion weights + the free-text projector |
| `models/blocks.npz` | 53.8 MB | the 8 embedding blocks (the hybrid embedding is fused from these) |
| `models/tfidf_baseline.joblib` | 21.2 MB | cache; without it the app refits TF-IDF on first run |
| `data/movies_clean.csv.gz` | 15.3 MB | the movie table |
| `models/ranker.joblib` | 10.8 MB | the trained XGBoost + stack |
| `models/app_meta.pkl`, `models/ok_mask.npy` | ~1.7 MB | row-order check, "never recommend" mask |

`.gitignore` deliberately still excludes `models/E_hybrid.npy` and `models/labels.csv` (the app
rebuilds/never reads them), `data/app.db`, `.venv/` and `__pycache__/`.

### Streamlit Community Cloud
1. `git init` in this folder, then `git add .` and commit (see the commands at the bottom).
2. Create an empty repo on GitHub and push:
   `git remote add origin https://github.com/<you>/cinematch.git` then `git push -u origin main`.
3. Go to **share.streamlit.io**, sign in with GitHub, **Create app**, pick the repo, branch `main`
   and main file `app.py`, then **Deploy**.
4. You get a shareable link like `https://<something>.streamlit.app`.

Two things that will bite you:
* **Versions are pinned in `requirements.txt`, and `.python-version` says 3.11, on purpose.**
  `models/ranker.joblib` is a pickle of scikit-learn/xgboost objects. If the host installs other
  versions, loading fails, the four notebook models are skipped (with a reason in the log) and the
  dropdown silently shows only `Baseline (TF-IDF)`.
* **The app must be started from the repo root** — `models/` and `data/` are resolved relative to
  the working directory. Streamlit Community Cloud does this for you; if you ever run it elsewhere,
  `cd` into the project first or the notebook models will not be found.

### Accounts and lists
`data/app.db` is local SQLite, so on a host **accounts, favorites and watched lists are wiped on
every restart or redeploy** (the container's disk is temporary). The app is fully usable as a guest
without it. For lasting data, swap `core/storage.py` — the only file that touches the DB — for a
hosted database such as Turso or Supabase.

### Local / demo laptop (recommended for a presentation)
`data/app.db` is a normal file there, so accounts and lists persist between runs.

### Other hosts
**Hugging Face Spaces** (free CPU: ~16 GB RAM, 50 GB disk, Git LFS supported) runs the same folder
with `sdk: streamlit` in the Space's README. **Render/Railway** need a Dockerfile but offer a
persistent disk, which would also fix the accounts problem.

```bash
# the whole deploy, from this folder
git init
git add .
git commit -m "CineMatch: baseline + notebook models + conversational chatbot"
git branch -M main
git remote add origin https://github.com/<you>/cinematch.git
git push -u origin main
```
