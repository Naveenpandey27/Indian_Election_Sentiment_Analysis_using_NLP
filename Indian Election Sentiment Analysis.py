# Import necessary libraries
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as ex

# Read data from CSV files
modi_data = pd.read_csv('modi_reviews.csv')
rahul_data = pd.read_csv('rahul_reviews.csv')

# Explore the Modi data
modi_data.head()
modi_data.shape

# Explore the Rahul data
rahul_data.head()
rahul_data.shape

# Text Cleaning

# Define a function for text preprocessing
def preprocess_modi_text_data(data, text_column_name):
    # Lowercasing
    data[text_column_name] = data[text_column_name].str.lower()

    # Check and handle missing values
    print(f"Data type of '{text_column_name}' column before handling missing values: {data[text_column_name].dtype}")
    print(f"Number of missing values in '{text_column_name}' column before handling missing values: {data[text_column_name].isnull().sum()}")
    data = data.dropna(subset=[text_column_name])

    # Convert the column to strings
    data.loc[:, text_column_name] = data[text_column_name].astype(str)

    # Remove special characters, emojis, and emoticons
    data.loc[:, text_column_name] = data[text_column_name].apply(lambda x: re.sub(r'[^a-zA-Z\s😀-🙁]', '', x))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    data.loc[:, text_column_name] = data[text_column_name].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

    # Stemming
    stemmer = PorterStemmer()
    data.loc[:, text_column_name] = data[text_column_name].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))

    # Remove numbers
    data.loc[:, text_column_name] = data[text_column_name].apply(lambda x: re.sub(r'\d+', '', x))

    return data

# Apply text preprocessing to Modi data
modi_data = preprocess_modi_text_data(modi_data, 'Tweet')
modi_data = modi_data.drop('Unnamed: 0', axis=1)

# Explore the cleaned Modi data
modi_data.head()

# Apply text preprocessing to Rahul data
def preprocess_rahul_text_data(data, text_column_name):
    # Lowercasing
    data[text_column_name] = data[text_column_name].str.lower()

    # Check and handle missing values
    print(f"Data type of '{text_column_name}' column before handling missing values: {data[text_column_name].dtype}")
    print(f"Number of missing values in '{text_column_name}' column before handling missing values: {data[text_column_name].isnull().sum()}")
    data = data.dropna(subset=[text_column_name])

    # Convert the column to strings
    data.loc[:, text_column_name] = data[text_column_name].astype(str)

    # Remove special characters, emojis, and emoticons
    data.loc[:, text_column_name] = data[text_column_name].apply(lambda x: re.sub(r'[^a-zA-Z\s😀-🙁]', '', x))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    data.loc[:, text_column_name] = data[text_column_name].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

    # Stemming
    stemmer = PorterStemmer()
    data.loc[:, text_column_name] = data[text_column_name].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))

    # Remove numbers
    data.loc[:, text_column_name] = data[text_column_name].apply(lambda x: re.sub(r'\d+', '', x))

    return data

rahul_data = preprocess_rahul_text_data(rahul_data, 'Tweet')
rahul_data = rahul_data.drop('Unnamed: 0', axis=1)

# Define a function to find the polarity of a review using TextBlob
def find_polarity(review):
    return TextBlob(review).sentiment.polarity

# Apply polarity analysis to Modi data
modi_data['Polarity'] = modi_data['Tweet'].apply(find_polarity)

# Apply polarity analysis to Rahul data
rahul_data['Polarity'] = rahul_data['Tweet'].apply(find_polarity)

# Assign sentiment labels based on polarity
modi_data['Label'] = np.where(modi_data['Polarity'] > 0, 'positive', 'negative')
modi_data['Label'][modi_data['Polarity'] == 0] = 'neutral'

rahul_data['Label'] = np.where(rahul_data['Polarity'] > 0, 'positive', 'negative')
rahul_data['Label'][rahul_data['Polarity'] == 0] = 'neutral'

# Remove neutral Modi reviews
neutral_modi_reviews = modi_data[modi_data['Polarity'] == 0.0000]
remove_modi_neutral_reviews = modi_data['Polarity'].isin(neutral_modi_reviews['Polarity'])
modi_data.drop(modi_data[remove_modi_neutral_reviews].index, inplace=True)

# Remove neutral Rahul reviews
neutral_rahul_reviews = rahul_data[rahul_data['Polarity'] == 0.0000]
remove_rahul_neutral_reviews = rahul_data['Polarity'].isin(neutral_rahul_reviews['Polarity'])
rahul_data.drop(rahul_data[remove_rahul_neutral_reviews].index, inplace=True)

# Randomly remove reviews to balance the dataset
np.random.seed(10)
remove_n_modi = 8481
drop_indices_modi = np.random.choice(modi_data.index, remove_n_modi, replace=True)
df_modi_data = modi_data.drop(drop_indices_modi)

np.random.seed(10)
remove_n_rahul = 360
drop_indices_rahul = np.random.choice(rahul_data.index, remove_n_rahul, replace=True)
df_rahul_data = rahul_data.drop(drop_indices_rahul)

# Prediction about Election

# Calculate the percentage of negative and positive reviews for Modi
modi_count = df_modi_data.groupby('Label').count()
neg_modi = (modi_count['Polarity'][0] / 1000) * 100
pos_modi = (modi_count['Polarity'][1] / 1000) * 100

# Calculate the percentage of negative and positive reviews for Rahul
rahul_count = df_rahul_data.groupby('Label').count()
neg_rahul = (rahul_count['Polarity'][0] / 1000) * 100
pos_rahul = (rahul_count['Polarity'][1] / 1000) * 100

# Create a bar chart to visualize sentiment analysis results
politicians = ['Narendra Modi', 'Rahul Gandhi']
neg_list = [neg_modi, neg_rahul]
pos_list = [pos_modi, pos_rahul]

fig = go.Figure(
    data=[
        go.Bar(name='Negative', x=politicians, y=neg_list, marker_color='red'),
        go.Bar(name='Positive', x=politicians, y=pos_list, marker_color='green')
    ]
)
fig.update_layout(barmode='group')
fig.show()
