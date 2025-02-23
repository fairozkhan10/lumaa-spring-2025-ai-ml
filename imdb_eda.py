import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # Optional: for interactive plots
import re
import warnings

# Configure display options and suppress warnings
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# ============================
# Step 1: Load the Dataset
# ============================
df = pd.read_csv('IMDB-Movie-Data.csv', nrows=500)

print("Dataset Preview (7 random rows):")
print(df.sample(7))
print("\nDataset shape:", df.shape)

# ============================
# Step 2: Basic EDA
# ============================
print("\nDataFrame Info:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe(include='all'))

print("\nMissing Values per Column:")
print(df.isnull().sum())

# Visualize missing values using a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.xlabel("Features")
plt.show()

# ============================
# Step 3: Data Cleaning
# ============================
# Remove any leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

print("\nColumns in Dataset:")
print(df.columns)

# Fill missing numeric values in 'Revenue (Millions)' and 'Metascore' with 0
df['Revenue (Millions)'] = pd.to_numeric(df['Revenue (Millions)'], errors='coerce')
df['Metascore'] = pd.to_numeric(df['Metascore'], errors='coerce')
df['Revenue (Millions)'].fillna(0, inplace=True)
df['Metascore'].fillna(0, inplace=True)

# For any remaining non-numeric missing values, fill with 'NAN'
df.fillna('NAN', inplace=True)

# ============================
# Step 4: Extracting Insights
# ============================

# Insight 1: IMDB Rating Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title("IMDB Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Insight 2: Runtime Distribution (in Minutes)
plt.figure(figsize=(10, 6))
sns.histplot(df['Runtime (Minutes)'], bins=20, kde=True)
plt.title("Runtime Distribution")
plt.xlabel("Runtime (Minutes)")
plt.ylabel("Count")
plt.show()

# Insight 3: Revenue vs Rating Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Revenue (Millions)', y='Rating', data=df)
plt.title("Revenue vs. Rating")
plt.xlabel("Revenue (Millions)")
plt.ylabel("Rating")
plt.show()

# Insight 4: Genre Analysis
genres = df['Genre'].str.split(',').explode().str.strip()
genre_counts = genres.value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title("Count of Movies by Genre")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
