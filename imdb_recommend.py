import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(csv_path, nrows=500):
    """
    Load the first nrows of the IMDB dataset and keep relevant columns:
    Title, Genre, Description, Year, Rating
    """
    df = pd.read_csv(csv_path, nrows=nrows)
    # Keep only relevant columns
    df = df[['Title', 'Genre', 'Description', 'Year', 'Rating']]
    
    # Drop rows with missing Title, Genre, or Description
    df.dropna(subset=['Title', 'Genre', 'Description'], inplace=True)
    
    # Fill missing Rating with 0 for potential re-ranking
    df['Rating'] = df['Rating'].fillna(0.0).astype(float)
    
    return df

def create_combined_text(df, genre_weight=2):
    """
    Combine Genre and Description into a single text field for TF-IDF.
    Repeats the Genre text 'genre_weight' times to emphasize it.
    """
    def repeat_genre(row):
        repeated_genre = (" " + row["Genre"])*genre_weight
        return repeated_genre + " " + row["Description"]
    
    # Create the CombinedText column
    df["CombinedText"] = df.apply(repeat_genre, axis=1)
    return df

def build_tfidf_matrix(df, text_column='CombinedText'):
    """
    Create the TF-IDF matrix from the specified text column.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[text_column].values.astype('U'))
    return tfidf, tfidf_matrix

def recommend(query, df, tfidf, tfidf_matrix, top_n=5, alpha=0.8):
    """
    Returns the top_n recommendations based on:
    final_score = alpha * similarity + (1 - alpha) * normalized_rating
    where normalized_rating = rating / 10.0.
    """
    # Transform the query into TF-IDF vector
    query_vec = tfidf.transform([query])
    
    # Cosine similarity (between query and each movie's combined text)
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Re-rank partially by rating
    normalized_ratings = df['Rating'] / 10.0
    final_scores = alpha * similarities + (1 - alpha) * normalized_ratings
    
    # Sort indices by final_scores in descending order
    sorted_indices = final_scores.argsort()[::-1]
    
    # Gather top_n results
    recommendations = []
    for idx in sorted_indices[:top_n]:
        recommendations.append({
            'Title': df.iloc[idx]['Title'],
            'Year': df.iloc[idx]['Year'],
            'Genre': df.iloc[idx]['Genre'],
            'Rating': df.iloc[idx]['Rating'],
            'Score': round(final_scores[idx], 4)
        })
    return recommendations

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced TF-IDF + Cosine Similarity Recommender for IMDB Movies."
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default="I love thrilling superhero action movies",
        help="User query describing desired movie attributes."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="IMDB-Movie-Data.csv",
        help="Path to the IMDB dataset CSV."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of recommendations to return."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Weight for text similarity (0 < alpha <= 1)."
    )
    parser.add_argument(
        "--genre_weight",
        type=int,
        default=2,
        help="Times to replicate the Genre text for emphasis."
    )
    
    args = parser.parse_args()
    
    # Load dataset
    df = load_dataset(args.csv_file)
    df = create_combined_text(df, genre_weight=args.genre_weight)
    
    # Build TF-IDF
    tfidf, tfidf_matrix = build_tfidf_matrix(df, text_column='CombinedText')
    
    # Recommend
    recommendations = recommend(args.query, df, tfidf, tfidf_matrix, top_n=args.top_n, alpha=args.alpha)
    
    # Print the results
    print(f"\nUser Query: {args.query}")
    print(f"Top {args.top_n} Recommendations (alpha={args.alpha}):\n")
    
    for i, rec in enumerate(recommendations, start=1):
        print(
            f"{i}) {rec['Title']} ({rec['Year']}), Genre: {rec['Genre']}, "
            f"IMDB Rating: {rec['Rating']}, Score: {rec['Score']}"
        )

if __name__ == "__main__":
    main()
