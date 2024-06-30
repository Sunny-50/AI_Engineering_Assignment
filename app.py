import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import os

st.title("Data Analysis App")

# Function to process the uploaded file
def process_uploaded_file(uploaded_file):
    data = []
    # Reading each line in the uploaded file
    for line in uploaded_file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")

    # Flatten the list of lists
    flat_data = []
    for item in data:
        if isinstance(item, list):
            for entry in item:
                if isinstance(entry, dict) and 'message' in entry:
                    try:
                        message_data = json.loads(entry['message'])
                        if isinstance(message_data, list):
                            for msg in message_data:
                                if isinstance(msg, dict):
                                    flat_data.append({
                                        'userid': entry.get('userid', ''),
                                        'timestamp': entry.get('timestamp', ''),
                                        'query': msg.get('query', ''),
                                        'response': msg.get('response', ''),
                                        'context': msg.get('context', 'NA'),
                                        'language': msg.get('language', 'AUTO'),
                                        'chatbotId': msg.get('chatbotId', 'NA'),
                                        'fromSource': entry.get('fromSource', ''),
                                        'queriesConsumed': entry.get('queriesConsumed', 0),
                                        'userType': entry.get('userType', ''),
                                        'modelType': entry.get('modelType', ''),
                                        'Version': entry.get('Version', '')
                                    })
                    except (json.JSONDecodeError, TypeError) as e:
                        st.error(f"Error processing message data: {e}")
                else:
                    st.error(f"Invalid entry structure: {entry}")
        else:
            st.error(f"Invalid item structure: {item}")

    # Convert the list of dictionaries to a DataFrame
    conversations = pd.DataFrame(flat_data)

    # Ensure 'query' and 'response' columns are string types
    conversations['query'] = conversations['query'].astype(str)
    conversations['response'] = conversations['response'].astype(str)

    # Combine 'query' and 'response' to form the 'text' column
    conversations['text'] = conversations['query'] + ' ' + conversations['response']

    # Perform clustering
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(conversations['text'])

    kmeans = KMeans(n_clusters=10, random_state=42)
    conversations['cluster'] = kmeans.fit_predict(X)

    # Extracting topics
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    topics = {}
    for i in range(10):
        topic_terms = [terms[ind] for ind in order_centroids[i, :10]]
        topics[i] = ' '.join(topic_terms)
    
    conversations['topic'] = conversations['cluster'].map(topics)

    # Perform sentiment analysis
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        score = analyzer.polarity_scores(text)
        if score['compound'] >= 0.05:
            return 'positive'
        elif score['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    conversations['sentiment'] = conversations['text'].apply(get_sentiment)

    # Save the processed data
    if not os.path.exists('./data'):
        os.makedirs('./data')
    conversations.to_csv('./data/processed_conversations.csv', index=False)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    joblib.dump(kmeans, './models/kmeans_model.pkl')

    return conversations

# File uploader
uploaded_file = st.file_uploader("Upload JSON file", type="json")

if uploaded_file is not None:
    conversations = process_uploaded_file(uploaded_file)

    # Display analysis
    st.header('Conversation Analysis')
    st.subheader('Counts')

    topic_counts = conversations['topic'].value_counts()
    sentiment_counts = conversations['sentiment'].value_counts()

    st.subheader('Topic Counts')
    st.table(topic_counts)

    st.subheader('Sentiment Counts')
    st.table(sentiment_counts)

    st.header('Sessions')
    page_size = 50
    page_number = st.number_input('Page Number', min_value=1, max_value=(len(conversations) // page_size) + 1, value=1)

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size

    st.table(conversations.iloc[start_idx:end_idx][['text', 'topic', 'sentiment']])
else:
    st.write("Please upload a JSON file.")
