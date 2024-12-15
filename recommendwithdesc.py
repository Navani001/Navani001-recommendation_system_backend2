import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
movies_data = pd.read_csv('./movieswithposter_updated.csv')
with open('model_vectorizer.pkl', 'rb') as vec_file, open('model_similarity.pkl', 'rb') as sim_file:
        vectorizer = pickle.load(vec_file)
        similarity = pickle.load(sim_file)
def recommend_movies_with_desc(query):
# Transform the query into a feature vector using the same vectorizer
  feature_vecto = vectorizer.transform(query)
  with open('feature_vector.pkl', 'rb') as feature:
        feature_vectors = pickle.load(feature)

  # Calculate cosine similarity between the query vector and the feature vectors of the movies
  sim = cosine_similarity(feature_vectors, feature_vecto)

  # Extract the similarity scores for the query against all movies
  combined_similarity = sim.flatten()

  # Sort the movies by similarity score
  sorted_similar_movies = list(enumerate(combined_similarity))
  sorted_similar_movies = sorted(sorted_similar_movies, key=lambda x: x[1], reverse=True)

  # Print out the top 5 similar movies
  i = 1
  movie_recom=[]
  for movie in sorted_similar_movies:
      index = movie[0]
#       title_from_index = movies_data.iloc[index]['title']  # Assuming movies_data is a DataFrame
      dataFromtitle = movies_data[movies_data.index == index]
      movie_recom.append({'title':dataFromtitle['title'].values[0],'image':dataFromtitle['poster'].values[0]})
      print(i, '.',dataFromtitle['title'].values[0], "(Score:", round(movie[1], 2), ")")
      i += 1
      if i > 35:  # Limit recommendations to top 5
          break
  return movie_recom