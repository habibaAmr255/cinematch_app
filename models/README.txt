Put the notebook's output files here (see the README, section "The models from the notebook").

Build them from the notebook's input with:

    python scripts/build_ml_models.py --data "path/to/df_sampled.csv"

Notebook models (hybrid embedding + XGBoost / stacked re-rankers):
  blocks.npz       required   the 8 embedding blocks (np.savez_compressed("blocks.npz", **blocks))
  ranker.joblib    required   {"stack", "xgb", "features", "feat_blocks"}
  fitted.joblib    optional   fusion weights + text-encoder spec (defaults = the notebook's)
  ok_mask.npy      optional   rows never recommended: duplicates / compilations (recomputed if missing)
  app_meta.pkl     optional   only used to check the rows line up with data/movies_clean.csv.gz
  E_hybrid.npy     optional   the fused embedding, kept for reuse (already offered as "Hybrid (cosine)")
  labels.csv       optional   "Title (Year)" labels, same row order

Any other embedding model: <name>.npy or <name>.npz (one vector per movie).
