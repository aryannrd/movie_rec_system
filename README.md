Added recommender.py, along with training.py and  app.py

recommender.py prepares the data further for the tf idf vectorizer which helps build the tf-idf model. following it up by using the sklit libray to create a cosine similarity using the tfidf model. We then use the cosine similiarity to get a score of each movie in the dataset and sort it to get the most similar movies to our movie. We then save the indexes of the movie along with their similarity score. Lastly, we also train model by calling all previous functions. For convinience, we save and load the model. 

For the training, we load the data and preprocess it and build our recommender. We then train it with respect to three colums: "overview", "genres" and "keywords". we then save the model. 

For app.py, we load the data again and pass it into our recommender function and load the model. then we pick a particular movie, for example interstellar, and print the results for it.
