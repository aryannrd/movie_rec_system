import sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class Recommender:
    def __init__(self, movie_df):
        if movie_df.empty:
            raise ValueError("Movies DataFrame is empty")

        self.movie_df= movie_df.copy()
        self.tfidf_vectorizer = None
        self.tfidf_model = None
        self.cosine_similarity = None
        self.indices= None

    def prepare_data(self, columns):
        if self.movie_df is None or self.movie_df.empty:
            raise ValueError("Movie DataFrame is empty")

        for col in columns:
            if col not in self.movie_df.columns:
                self.movie_df[col] = ""

        self.movie_df["combined_columns"] = (
            self.movie_df[columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )

        self.movie_df["combined_columns"] = (
            self.movie_df["combined_columns"]
            .fillna("")
        )

    def build_tfidf_model(self, max_features=5000, ngram_range=(1,2), min_df=2):
        if 'combined_columns' not in self.movie_df.columns:
            raise ValueError("Combined columns are missing")
        print("Initializing tfidf model")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, stop_words='english')
        self.tfidf_model= self.tfidf_vectorizer.fit_transform(self.movie_df["combined_columns"])

        print(f"Tfidf model shape: {self.tfidf_model.shape}")

    def similarity_model(self):

        if self.tfidf_model is None:
            raise ValueError("tfidf model is empty")
        self.cosine_similarity = cosine_similarity(self.tfidf_model, self.tfidf_model)
        self.movie_df["title_norm"] = (
            self.movie_df["title"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        print(f"Cosine similarity shape: {self.cosine_similarity.shape}")

        self.indices = pd.Series(
            self.movie_df.index,
            index=self.movie_df["title_norm"]
        ).drop_duplicates()

    def recommend_movies(self, n_rec, m_title):
        if self.cosine_similarity is None:
            raise ValueError("Model not trained")

        title_key = str(m_title).strip().lower()

        if title_key not in self.indices:
            raise KeyError(f"Movie '{m_title}' not found")

        idx = self.indices[title_key]

        sim_scores = list(enumerate(self.cosine_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_rec + 1]

        movie_indices = [i[0] for i in sim_scores]
        similarities = [i[1] for i in sim_scores]

        recs = self.movie_df.iloc[movie_indices].copy()
        recs["similarity"] = similarities

        return recs

    def get_recommendations_by_id(self, movie_id, n_rec=10):
        movie_row = self.movie_df[self.movie_df["id"] == movie_id]
        if movie_row.empty:
            return f"Movie ID '{movie_id}' not found"

        movie_title = movie_row.iloc[0]["title"]
        return self.recommend_movies(n_rec, movie_title)

    def train_recommender(self, feature_columns, max_features=5000, ngram_range=(1, 2), min_df=2):
        print("Starting recommender training...")

        self.prepare_data(feature_columns)
        self.build_tfidf_model(max_features, ngram_range, min_df)
        self.similarity_model()
        print("Training complete!")

    def save_model(self, filepath='models/recommender_model.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_model': self.tfidf_model,
            'cosine_similarity': self.cosine_similarity,
            'indices': self.indices,
            'movie_df': self.movie_df
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath='models/recommender_model.pkl'):

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_model = model_data['tfidf_model']
        self.cosine_similarity = model_data['cosine_similarity']
        self.indices = model_data['indices']
        self.movie_df = model_data['movie_df']

        print(f"Model loaded from {filepath}")

