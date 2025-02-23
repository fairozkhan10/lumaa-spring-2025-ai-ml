# Simple Content-Based Movie Recommender

## Overview
This project builds a content-based movie recommender using TF-IDF and cosine similarity, with the following enhancements:
- **imdb_eda.py**: A script that performs exploratory data analysis (EDA) on the first 500 rows of the `IMDB-Movie-Data.csv` dataset.
- **imdb_recommend.py**: An enhanced recommendation system that:
  1. Combines Genre and Description for richer text context.
  2. Applies a simple rating-based re-rank to boost higher-rated movies.
  3. Provides a command-line interface (CLI) with arguments for query, top_n, alpha, and genre_weight.

This project was submitted as part of the AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation.

## Example Use Case
- **User Query**: "I love thrilling action movies set in space, with a comedic twist."
- **System Output**: The system processes this query, compares it against a dataset of movies, and returns the top 3–5 closest matches.

## Requirements
1. **Dataset**
   - **IMDB-Movie-Data.csv** (500 rows).
   - Source: [IMDB-Movie-Data.csv on GitHub](https://github.com/LearnDataSci/articles/blob/master/Python%20Pandas%20Tutorial%20A%20Complete%20Introduction%20for%20Beginners/IMDB-Movie-Data.csv).
   - The dataset is included in this repository.

2. **Approach**
   - Use TF-IDF to convert text (combined Genre and Description) into vectors.
   - Compute cosine similarity between a user’s query and each movie’s text.
   - Optionally, re-rank results using the movie’s IMDB Rating.
   - Return the top N recommendations (default 5).

3. **Code Organization**
   - Python scripts: `imdb_eda.py` for EDA and `imdb_recommend.py` for recommendations.
   - Functions are modular and include docstrings for clarity.

4. **Output**
   - When a user inputs a query, the system prints a list of recommended movie titles along with similarity scores and additional details (Year, Genre, IMDB Rating).

## Setup
1. **Python Version**: 3.9+ (or similar).
2. **Virtual Environment** (Recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
