import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')

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
    try:
        raw_data = uploaded_file.read().decode('utf-8')
        data = json.loads(f"[{raw_data.replace('}{', '},{')}]")
    except json.JSONDecodeError:
        st.error("Error decoding JSON. Please check the file format.")
        st.stop()

    # Flatten the JSON
    messages = []
    for item in data:
        try:
            msg = json.loads(item['message'])
            messages.append(msg)
        except (json.JSONDecodeError, KeyError):
            continue

    df = pd.json_normalize(messages)

    # Assuming the JSON has 'value' field for analysis
    if 'value' in df.columns:
        # Vectorize the text data
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        text_vectorized = vectorizer.fit_transform(df['value'].dropna().tolist())

        # Apply LDA for topic modeling
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(text_vectorized)

        # Assign topics to each text
        topic_results = lda.transform(text_vectorized)
        df['topic'] = topic_results.argmax(axis=1)
        df['topic'] = df['topic'].apply(lambda x: f"Topic {x+1}")

        # Apply sentiment analysis
        df['sentiment'] = df['value'].apply(get_sentiment)

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

        st.table(df.iloc[start_idx:end_idx][['value', 'topic', 'sentiment']])
    else:
        st.write("No 'value' field found in the uploaded JSON file.")
