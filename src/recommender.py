"""
RankFlix Recommender - Production Inference Module
Two-stage hybrid recommender: Candidate Generation + LightGBM Ranking
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import lightgbm as lgb
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """
    RankFlix Recommender.
    
    Stage 1: Candidate generation (popularity + tag-based semantic)
    Stage 2: LightGBM LambdaRank scoring
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all pre-trained artifacts from disk."""
        # LightGBM ranker (native loading)
        self.ranker = lgb.Booster(model_file=str(self.models_dir / "lgbm_ranker.txt"))
        
        # TF-IDF vectorizer for tag transformation
        with open(self.models_dir / "tfidf.pkl", "rb") as f:
            self.tfidf = pickle.load(f)
        
        # Movie-tag TF-IDF matrix (sparse)
        self.movie_tag_matrix = load_npz(self.models_dir / "movie_tag_matrix.npz")
        
        # Popularity scores
        with open(self.models_dir / "popularity.pkl", "rb") as f:
            self.popularity = pickle.load(f)
        
        # ID mappings
        with open(self.models_dir / "movie_id_to_idx.pkl", "rb") as f:
            self.movie_id_to_idx = pickle.load(f)
        
        with open(self.models_dir / "idx_to_movie_id.pkl", "rb") as f:
            self.idx_to_movie_id = pickle.load(f)
        
        # Precompute sorted popularity for fast candidate retrieval
        self._popularity_sorted = sorted(
            self.popularity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Valid movie IDs set for filtering
        self._valid_movie_ids = set(self.movie_id_to_idx.keys())
    
    # =========================================================================
    # User Profile Construction
    # =========================================================================
    
    def build_profile_from_tags(self, tags: list[str]) -> np.ndarray:
        """
        Build user profile from input tags using TF-IDF transform.
        Returns a 1D dense array representing the user's tag preferences.
        """
        if not tags:
            return np.zeros(self.movie_tag_matrix.shape[1])
        
        tag_text = " ".join(tags)
        profile = self.tfidf.transform([tag_text])
        return profile.toarray().flatten()
    
    def build_profile_from_movies(self, movie_ids: list[int]) -> np.ndarray:
        """
        Build user profile from liked movies by averaging their tag vectors.
        """
        valid_ids = [mid for mid in movie_ids if mid in self.movie_id_to_idx]
        
        if not valid_ids:
            return np.zeros(self.movie_tag_matrix.shape[1])
        
        indices = [self.movie_id_to_idx[mid] for mid in valid_ids]
        movie_vectors = self.movie_tag_matrix[indices].toarray()
        profile = np.mean(movie_vectors, axis=0)
        return profile
    
    def build_profile(self, input_data: dict) -> Optional[np.ndarray]:
        """
        Build user profile based on available input data.
        Returns None for anonymous users (popularity fallback).
        """
        # Priority: liked_tags > liked_movies > recent_movies > anonymous
        if "liked_tags" in input_data and input_data["liked_tags"]:
            return self.build_profile_from_tags(input_data["liked_tags"])
        
        if "liked_movies" in input_data and input_data["liked_movies"]:
            return self.build_profile_from_movies(input_data["liked_movies"])
        
        if "recent_movies" in input_data and input_data["recent_movies"]:
            return self.build_profile_from_movies(input_data["recent_movies"])
        
        # Anonymous user or userId without profile data
        return None
    
    # =========================================================================
    # Candidate Generation (Stage 1)
    # =========================================================================
    
    def popularity_candidates(self, k: int) -> list[int]:
        """
        Return top-K movies by popularity score.
        """
        return [movie_id for movie_id, _ in self._popularity_sorted[:k]]
    
    def tag_candidates(self, user_profile: np.ndarray, k: int) -> list[int]:
        """
        Return top-K movies by tag similarity to user profile.
        Uses cosine similarity between user profile and movie tag vectors.
        """
        if user_profile is None or np.allclose(user_profile, 0):
            return []
        
        # Compute cosine similarity (user_profile vs all movies)
        user_profile_2d = user_profile.reshape(1, -1)
        similarities = cosine_similarity(user_profile_2d, self.movie_tag_matrix).flatten()
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Convert indices to movie IDs
        candidates = [
            self.idx_to_movie_id[idx] 
            for idx in top_indices 
            if idx in self.idx_to_movie_id
        ]
        return candidates
    
    def generate_candidates(
        self, 
        user_profile: Optional[np.ndarray], 
        k_pop: int = 100, 
        k_tag: int = 100
    ) -> list[int]:
        """
        Generate deduplicated candidate set from popularity and tag-based sources.
        """
        pop_candidates = self.popularity_candidates(k_pop)
        
        if user_profile is not None:
            tag_cands = self.tag_candidates(user_profile, k_tag)
        else:
            tag_cands = []
        
        # Deduplicate while preserving order (popularity first)
        seen = set()
        candidates = []
        
        for mid in pop_candidates + tag_cands:
            if mid not in seen:
                seen.add(mid)
                candidates.append(mid)
        
        return candidates
    
    # =========================================================================
    # Feature Extraction
    # =========================================================================
    
    def extract_features(
        self, 
        user_profile: Optional[np.ndarray], 
        movie_id: int
    ) -> np.ndarray:
        """
        Extract features for a (user, movie) pair.
        
        Features:
        - popularity: normalized popularity score
        - log_popularity: log-transformed popularity
        - tag_similarity: cosine similarity between user and movie
        """
        # Popularity features
        pop_score = self.popularity.get(movie_id, 0.0)
        log_pop = np.log1p(pop_score)
        
        # Tag similarity
        if user_profile is not None and movie_id in self.movie_id_to_idx:
            idx = self.movie_id_to_idx[movie_id]
            movie_vector = self.movie_tag_matrix[idx].toarray().flatten()
            
            user_norm = np.linalg.norm(user_profile)
            movie_norm = np.linalg.norm(movie_vector)
            
            if user_norm > 0 and movie_norm > 0:
                tag_sim = np.dot(user_profile, movie_vector) / (user_norm * movie_norm)
            else:
                tag_sim = 0.0
        else:
            tag_sim = 0.0
        
        return np.array([pop_score, log_pop, tag_sim], dtype=np.float32)
    
    def extract_features_batch(
        self, 
        user_profile: Optional[np.ndarray], 
        movie_ids: list[int]
    ) -> np.ndarray:
        """
        Extract features for multiple movies efficiently.
        Returns shape (n_movies, n_features).
        """
        n_movies = len(movie_ids)
        features = np.zeros((n_movies, 3), dtype=np.float32)
        
        for i, mid in enumerate(movie_ids):
            features[i] = self.extract_features(user_profile, mid)
        
        return features
    
    # =========================================================================
    # Ranking (Stage 2)
    # =========================================================================
    
    def rank_candidates(
        self, 
        user_profile: Optional[np.ndarray], 
        candidates: list[int], 
        k: int = 10
    ) -> list[int]:
        """
        Rank candidate movies using LightGBM ranker.
        Returns top-K movie IDs sorted by predicted score.
        """
        if not candidates:
            return []
        
        # Extract features for all candidates
        features = self.extract_features_batch(user_profile, candidates)
        
        # Predict scores
        scores = self.ranker.predict(features)
        
        # Sort by score descending and return top-K
        sorted_indices = np.argsort(scores)[::-1][:k]
        ranked_movies = [candidates[i] for i in sorted_indices]
        
        return ranked_movies
    
    # =========================================================================
    # Unified Online Pipeline
    # =========================================================================
    
    def recommend_online(
        self, 
        input_data: dict, 
        k: int = 10,
        k_pop: int = 100,
        k_tag: int = 100
    ) -> list[int]:
        """
        Main recommendation entry point for online inference.
        
        Args:
            input_data: User context dictionary. Supports:
                - {} → anonymous user (popularity fallback)
                - {"userId": int} → known user (requires cached profile)
                - {"liked_tags": [str, ...]} → cold-start with tags
                - {"liked_movies": [int, ...]} → cold-start with movies
                - {"recent_movies": [int, ...]} → session-based
            k: Number of recommendations to return
            k_pop: Number of popularity candidates
            k_tag: Number of tag-based candidates
        
        Returns:
            List of recommended movie IDs
        """
        # Build user profile
        user_profile = self.build_profile(input_data)
        
        # Generate candidates
        candidates = self.generate_candidates(user_profile, k_pop, k_tag)
        
        if not candidates:
            # Fallback to pure popularity
            return self.popularity_candidates(k)
        
        # Rank and return top-K
        recommendations = self.rank_candidates(user_profile, candidates, k)
        
        return recommendations
    
    def recommend_for_anonymous(self, k: int = 10) -> list[int]:
        """Convenience method for anonymous users."""
        return self.recommend_online({}, k=k)
    
    def recommend_for_tags(self, tags: list[str], k: int = 10) -> list[int]:
        """Convenience method for tag-based cold start."""
        return self.recommend_online({"liked_tags": tags}, k=k)
    
    def recommend_for_movies(self, movie_ids: list[int], k: int = 10) -> list[int]:
        """Convenience method for movie-based cold start."""
        return self.recommend_online({"liked_movies": movie_ids}, k=k)


# Module-level instance for Streamlit import convenience
_recommender_instance: Optional[MovieRecommender] = None


def get_recommender(models_dir: str = "models") -> MovieRecommender:
    """
    Get or create the recommender singleton.
    Use this in Streamlit to avoid reloading on each interaction.
    """
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = MovieRecommender(models_dir)
    return _recommender_instance


def recommend(input_data: dict, k: int = 10, models_dir: str = "models") -> list[int]:
    """
    Functional interface for recommendations.
    
    Example usage in Streamlit:
        from recommender import recommend
        movies = recommend({"liked_tags": ["sci-fi", "action"]}, k=10)
    """
    recommender = get_recommender(models_dir)
    return recommender.recommend_online(input_data, k=k)
