from data_loader import load_movie_data
from recommender import Recommender

df = load_movie_data("/Users/aryan/movie_rec_system/data/raw/movies.csv")

recommender = Recommender(df)
recommender.load_model("/Users/aryan/movie_rec_system/src/models/recommender_model.pkl")

results = recommender.recommend_movies(
    n_rec=5,
    m_title="Interstellar"
)

if isinstance(results, str):
    print(results)   # error message
else:
    print(results[["title", "similarity"]])