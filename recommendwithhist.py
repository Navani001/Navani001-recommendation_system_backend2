import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
def stemmed_tokenizer(text):
    ps = PorterStemmer()
    words = word_tokenize(text)
    return [ps.stem(word) for word in words]
# Initialize an empty dictionary to store user history
user_history = {}

# Function to save user history to a pickle file
def save_user_history():
    with open('user_history.pkl', 'wb') as file:
        pickle.dump(user_history, file)

# Function to load user history from a pickle file
def load_user_history():
    global user_history
    if os.path.exists('user_history.pkl'):
        with open('user_history.pkl', 'rb') as file:
            user_history = pickle.load(file)

# Load movie data
movies_data = pd.read_csv('./movieswithposter_updated.csv')

# Pre-process data
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Check if the model (vectorizer and similarity) exists
if os.path.exists('model_vectorizer.pkl') and os.path.exists('model_similarity.pkl'):
    # Load the vectorizer and similarity matrix
    with open('model_vectorizer.pkl', 'rb') as vec_file, open('model_similarity.pkl', 'rb') as sim_file:
        vectorizer = pickle.load(vec_file)
        similarity = pickle.load(sim_file)
else:
    # Train the model if it doesn't exist
    vectorizer = TfidfVectorizer(stop_words='english',tokenizer=stemmed_tokenizer)
    feature_vectors = vectorizer.fit_transform(combined_features)
    with open('feature_vector.pkl', 'wb') as file:
        pickle.dump(feature_vectors, file)
    # Calculate cosine similarity
    similarity = cosine_similarity(feature_vectors)

    # Save the model (vectorizer and similarity matrix)
    with open('model_vectorizer.pkl', 'wb') as vec_file, open('model_similarity.pkl', 'wb') as sim_file:
        pickle.dump(vectorizer, vec_file)
        pickle.dump(similarity, sim_file)

# Function to recommend movies based on both user input and history
def recommend_movieswithhistory(user_id, movie_name):
    # Add the movie to the user's history
    add_to_history(user_id, movie_name)
    print(user_id,movie_name)
    # Fetch the user's history
    history = get_history(user_id)
    
    if len(history) == 0:
        print("No history found for the user.")
        return

    print(f"Movies suggested for you based on your past choices: {history}\n")

    # Create an aggregate similarity score across all movies in history
    combined_similarity = np.zeros(similarity.shape[0])
    
    for past_movie in history:
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
    movie_return=[]
    for movie in sorted_similar_movies:
        index = movie[0]
        # title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        dataFromtitle = movies_data[movies_data.index == index]
        
        
        if dataFromtitle['title'].values[0] not in history:  # Don't recommend movies the user has already interacted with
            
            print(i, '.',dataFromtitle['title'].values[0], "(Score:", round(movie[1], 2), ")")
            movie_return.append({'title':dataFromtitle['title'].values[0],'image':dataFromtitle['poster'].values[0]})
            i += 1
            if i > 35:  # Limit recommendations to top 5
                break
    return movie_return

# Function to add a movie to user history
def add_to_history(user_id, movie_title):
    if user_id not in user_history:
        user_history[user_id] = []
    user_history[user_id].append(movie_title)
    save_user_history()  # Save the updated history after adding a movie

# Function to get movies from user history
def get_history(user_id):
    return user_history.get(user_id, [])

# Load the user history at the start of the program
load_user_history()


