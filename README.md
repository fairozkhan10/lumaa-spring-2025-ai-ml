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

 **Example Output**
User Query: I love thrilling superhero action movies
Top 5 Recommendations (alpha=0.8):

1) Kick-Ass (2010), Genre: Action,Comedy, IMDB Rating: 7.7, Score: 0.3681
2) Let Me Make You a Martyr (2016), Genre: Action,Crime,Drama, IMDB Rating: 6.4, Score: 0.2497
3) The Longest Ride (2015), Genre: Drama,Romance, IMDB Rating: 7.1, Score: 0.2450
4) 20th Century Women (2016), Genre: Comedy,Drama, IMDB Rating: 7.4, Score: 0.2383
5) Jagten (2012), Genre: Drama, IMDB Rating: 8.3, Score: 0.2264


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



## My salary expectation is anything between $25-$35/hr. 


## Link to the Short Video Demo via Kaltura Media Space (UW-Madison)- https://mediaspace.wisc.edu/media/Screen+Recording+2025-02-23+at+4.51.22%E2%80%AFPM/1_m76yexdt
