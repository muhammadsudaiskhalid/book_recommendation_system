from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

app = Flask(__name__)

# Global variables for model components
model_data = None

def load_model():
    """Load the trained recommendation model"""
    global model_data
    
    model_paths = ['book_recommendation_model.pkl', 'model.pkl']
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                print(f"[INFO] Model loaded successfully from {model_path}")
                
                # Validate model data structure
                required_keys = ['knn_model', 'scaler', 'books_data']
                if all(key in model_data for key in required_keys):
                    print(f"[INFO] Model contains {len(model_data['books_data'])} books")
                    return True
                else:
                    print(f"[WARNING] Model from {model_path} missing required components")
                    
            except Exception as e:
                print(f"[ERROR] Could not load model from {model_path}: {e}")
    
    print("[ERROR] No valid model found")
    return False

def search_books_by_text(query, top_k=10):
    """Search books using text similarity"""
    try:
        if 'tfidf' not in model_data or 'tfidf_matrix' not in model_data:
            # Fallback to simple text search
            books_df = model_data['books_data']
            query_lower = query.lower()
            
            # Search in title and authors
            title_matches = books_df['title'].str.lower().str.contains(query_lower, na=False)
            author_matches = books_df['authors'].str.lower().str.contains(query_lower, na=False)
            
            matches = books_df[title_matches | author_matches]
            return matches.head(top_k)
        
        # Use TF-IDF for better text matching
        tfidf = model_data['tfidf']
        tfidf_matrix = model_data['tfidf_matrix']
        
        query_vec = tfidf.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        books_df = model_data['books_data']
        return books_df.iloc[top_indices]
        
    except Exception as e:
        print(f"[ERROR] Error in text search: {e}")
        return pd.DataFrame()

def get_similar_books(book_index, top_k=5):
    """Get similar books using KNN model"""
    try:
        features = model_data['features']
        scaler = model_data['scaler']
        knn_model = model_data['knn_model']
        
        # Get features for the selected book
        book_features = features.iloc[book_index:book_index+1]
        book_features_scaled = scaler.transform(book_features)
        
        # Find similar books
        distances, indices = knn_model.kneighbors(book_features_scaled, n_neighbors=top_k+1)
        
        # Remove the book itself from recommendations
        similar_indices = [idx for idx in indices[0] if idx != book_index][:top_k]
        
        books_df = model_data['books_data']
        return books_df.iloc[similar_indices]
        
    except Exception as e:
        print(f"[ERROR] Error getting similar books: {e}")
        return pd.DataFrame()

def filter_books(books_df, language=None, min_rating=None):
    """Filter books based on criteria"""
    try:
        filtered_df = books_df.copy()
        
        if language and language != '':
            filtered_df = filtered_df[filtered_df['language_code'] == language]
        
        if min_rating and min_rating != '':
            min_rating = float(min_rating)
            filtered_df = filtered_df[filtered_df['average_rating'] >= min_rating]
        
        return filtered_df
        
    except Exception as e:
        print(f"[ERROR] Error filtering books: {e}")
        return books_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if model_data is None:
        return render_template('index.html', 
                             error_message='Model not loaded. Please check the server configuration.')

    try:
        # Get form inputs
        book_title = request.form.get('book_title', '').strip()
        language = request.form.get('language', '')
        min_rating = request.form.get('min_rating', '')
        
        if not book_title:
            return render_template('index.html', 
                                 error_message='Please enter a book title or author name.')
        
        # Search for books matching the query
        search_results = search_books_by_text(book_title, top_k=20)
        
        if search_results.empty:
            return render_template('index.html', 
                                 error_message=f'No books found matching "{book_title}". Try a different search term.')
        
        # Filter results based on user preferences
        filtered_results = filter_books(search_results, language, min_rating)
        
        if filtered_results.empty:
            return render_template('index.html', 
                                 error_message='No books match your criteria. Try adjusting your filters.')
        
        # Get the best match and find similar books
        best_match_index = filtered_results.index[0]
        similar_books = get_similar_books(best_match_index, top_k=5)
        
        # Combine and deduplicate recommendations
        all_recommendations = pd.concat([filtered_results.head(3), similar_books]).drop_duplicates(subset=['title'])
        
        # Apply final filtering
        final_recommendations = filter_books(all_recommendations, language, min_rating)
        
        # Prepare recommendations for template
        recommendations = []
        for _, book in final_recommendations.head(8).iterrows():
            recommendations.append({
                'title': book['title'],
                'authors': book['authors'],
                'average_rating': round(float(book['average_rating']), 1),
                'ratings_count': int(book['ratings_count'])
            })
        
        if not recommendations:
            return render_template('index.html', 
                                 error_message='No recommendations found matching your criteria.')
        
        return render_template('index.html', recommendations=recommendations)
        
    except Exception as e:
        print(f"[ERROR] Error in recommend route: {e}")
        return render_template('index.html', 
                             error_message=f'An error occurred while processing your request: {str(e)}')

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', 
                         error_message='Page not found.'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', 
                         error_message='Internal server error. Please try again.'), 500

if __name__ == "__main__":
    print("Starting Book Recommendation System...")
    
    # Load the model
    if not load_model():
        print("[ERROR] Failed to load model. Please run model.py first to train the model.")
        print("The application will start but recommendations will not work.")
    
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)