# import pandas as pd
# from datasets import load_dataset
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import os
# import joblib

# os.environ["LOKY_MAX_CPU_COUNT"] = "2" 

# dataset = load_dataset("LDJnr/Puffin")

# conversations = pd.DataFrame(dataset['train'])

# conversations['text'] = conversations['conversations'].apply(lambda x: " ".join([turn['value'] for turn in x]))

# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(conversations['text'])

# kmeans = KMeans(n_clusters=10, random_state=42)
# conversations['cluster'] = kmeans.fit_predict(X)

# topics = {0: 'Programming Practices', 1: 'Customer Support'}  
# conversations['topic'] = conversations['cluster'].map(topics).fillna('Misc')

# analyzer = SentimentIntensityAnalyzer()

# def get_sentiment(text):
#     score = analyzer.polarity_scores(text)
#     if score['compound'] >= 0.05:
#         return 'positive'
#     elif score['compound'] <= -0.05:
#         return 'negative'
#     else:
#         return 'neutral'

# conversations['sentiment'] = conversations['text'].apply(get_sentiment)

# if not os.path.exists('./data'):
#     os.makedirs('./data')
# conversations.to_csv('./data/processed_conversations.csv', index=False)

# if not os.path.exists('./models'):
#     os.makedirs('./models')
# joblib.dump(kmeans, './models/kmeans_model.pkl')

import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Title of the app
st.title("Conversation Analysis App")

# File uploader section
uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

if uploaded_file is not None:
    # Read and process JSON file
    data = json.load(uploaded_file)
    df = pd.json_normalize(data)

    # Assuming the JSON has 'text' field for analysis
    if 'text' in df.columns:
        # Vectorize the text data
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        text_vectorized = vectorizer.fit_transform(df['text'].dropna().tolist())

        # Apply LDA for topic modeling
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(text_vectorized)

        # Assign topics to each text
        topic_results = lda.transform(text_vectorized)
        df['topic'] = topic_results.argmax(axis=1)
        df['topic'] = df['topic'].apply(lambda x: f"Topic {x+1}")

        # Apply sentiment analysis
        df['sentiment'] = df['text'].apply(get_sentiment)

        # Screen 1: Counts
        st.header('Counts')

        topic_counts = df['topic'].value_counts()
        sentiment_counts = df['sentiment'].value_counts()

        st.subheader('Topic Counts')
        st.table(topic_counts)

        st.subheader('Sentiment Counts')
        st.table(sentiment_counts)

        # Screen 2: Sessions
        st.header('Sessions')
        page_size = 50
        page_number = st.number_input('Page Number', min_value=1, max_value=(len(df) // page_size) + 1, value=1)

        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size

        st.table(df.iloc[start_idx:end_idx][['text', 'topic', 'sentiment']])
    else:
        st.write("No 'text' field found in the uploaded JSON file.")
