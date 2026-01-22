from data_loader import load_movie_data
from preprocessing import preprocess
from recommender import Recommender

path = "/Users/aryan/movie_rec_system/data/raw/movies.csv"

FEATURE_COLUMNS = [
    "overview",
    "genres",
    "keywords",
]

def main():
    df = load_movie_data(path)
    df = preprocess(df)
    recommender = Recommender(df)
    recommender.train_recommender(
        feature_columns=FEATURE_COLUMNS,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    recommender.save_model("models/recommender_model.pkl")

if __name__ == "__main__":
    main()
