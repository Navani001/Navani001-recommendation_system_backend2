import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
import difflib
movies_data = pd.read_csv('./movieswithposter_updated.csv')
def recommend_movies(movie_name):
    # Add the movie to the user's history
    if os.path.exists('model_vectorizer.pkl') and os.path.exists('model_similarity.pkl'):
    # Load the vectorizer and similarity matrix
        with open('model_vectorizer.pkl', 'rb') as vec_file, open('model_similarity.pkl', 'rb') as sim_file:
            vectorizer = pickle.load(vec_file)
            similarity = pickle.load(sim_file)

    print(f"Movies suggested for you based on your past choices: \n")

    # Create an aggregate similarity score across all movies in history
    combined_similarity = np.zeros(similarity.shape[0])
    
    for past_movie in [movie_name]:
        # Find a close match for each movie in the user's history
        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(past_movie, list_of_all_titles)

        if find_close_match:
            close_match = find_close_match[0]
            # Find the index of the movie in the dataset
            index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
            # Accumulate the similarity scores
            combined_similarity += similarity[index_of_the_movie]

    # Sort movies based on the combined similarity score
    sorted_similar_movies = list(enumerate(combined_similarity))
    sorted_similar_movies = sorted(sorted_similar_movies, key=lambda x: x[1], reverse=True)

    # Recommend the top movies that the user hasn't already seen
    i = 1
    movie_returns = []
    for movie in sorted_similar_movies:
        index = movie[0]
        # title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        dataFromtitle = movies_data[movies_data.index == index]
        movie_returns.append({'title':dataFromtitle['title'].values[0],'image':dataFromtitle['poster'].values[0]})
        print(i, '.',dataFromtitle['title'].values[0], "(Score:", round(movie[1], 2), ")")
        
        i+=1
        if i > 35:  # Limit recommendations to top 5
                break
    return movie_returns
        