# RankFlix Recommender

> Two-stage hybrid recommendation system with candidate generation + learning-to-rank  
> Built with Python, TF-IDF tag semantics, and LightGBM LambdaRank

------------------------------------------------------------------------

## Overview

RankFlix Recommender is an end-to-end movie recommendation project that includes:

- **Training notebooks** (MovieLens) to build recommendation signals
- **Reusable inference module** for online recommendations
- **Streamlit demo app** for interactive exploration

The core idea is simple and production-friendly:

1) Generate a small set of good candidates quickly  
2) Rank them with a trained learning-to-rank model

------------------------------------------------------------------------

## Architecture

    User Input (anonymous / tags / liked movies)
                 ↓
    Candidate Generation (Popularity + Tag Similarity)
                 ↓
        Feature Extraction (popularity, log-pop, tag_sim)
                 ↓
      LightGBM LambdaRank (learning-to-rank scoring)
                 ↓
            Top-K Recommendations

### Online modes supported

- **Anonymous**: popularity-only fallback
- **By tags**: cold-start personalization (e.g. “sci-fi”, “thriller”)
- **By liked movies**: profile from a small set of favorites

------------------------------------------------------------------------

## Models & Artifacts

Precomputed artifacts live in `models/`:

- `lgbm_ranker.txt`  
- `tfidf.pkl`  
- `movie_tag_matrix.npz`  
- `popularity.pkl`  
- `movie_id_to_idx.pkl`, `idx_to_movie_id.pkl`

These are loaded once and reused for fast inference.

------------------------------------------------------------------------

## Project Structure (recommended after rename)

    rankflix-recommender/
    │
    ├── app/                     # Streamlit demo
    │   └── app.py
    │
    ├── src/                     # Reusable inference module
    │   └── recommender.py
    │
    ├── notebooks/               # Training + experiments
    │   ├── recommender_baseline.ipynb
    │   ├── recommender_movielens_10m.ipynb
    │   └── recommender_proposed_method.ipynb
    │
    ├── data/                    # MovieLens datasets (raw)
    │   └── movielens/
    │
    ├── models/                  # Trained artifacts for inference
    │   └── ...
    │
    └── README.md

> Your current repo already contains these pieces, but notebooks and datasets are in the root.
> The layout above is the “product-style” structure.

------------------------------------------------------------------------

## Run the demo (Streamlit)

Install dependencies:

```bash
pip install -U streamlit pandas numpy scikit-learn scipy lightgbm
```

Start the app:

```bash
streamlit run app/app.py
```

If you keep the current layout, run:

```bash
streamlit run app.py
```

------------------------------------------------------------------------

## Using the recommender as a module

```python
from src.recommender import recommend

movies = recommend({"liked_tags": ["sci-fi", "action"]}, k=10, models_dir="models")
print(movies)
```

Input examples:

- `{}` (anonymous)
- `{"liked_tags": ["sci-fi", "thriller"]}`
- `{"liked_movies": [1, 260, 1196]}`

------------------------------------------------------------------------

## Why this approach works in real products

- **Fast** candidate generation (cheap similarity + popularity)
- **Better ranking** using learning-to-rank (LambdaRank)
- **Cold-start friendly** via tags or a few liked items
- **Artifact-based inference**: deployable without notebooks

------------------------------------------------------------------------

## License

MIT License
