import streamlit as st
import pandas as pd
import pickle

# Load the processed data
conversations = pd.read_csv('./data/processed_conversations.csv')

# Screen 1: Counts
st.title('Conversation Analysis')
st.header('Counts')

topic_counts = conversations['topic'].value_counts()
sentiment_counts = conversations['sentiment'].value_counts()

st.subheader('Topic Counts')
st.table(topic_counts)

st.subheader('Sentiment Counts')
st.table(sentiment_counts)

# Screen 2: Sessions
st.header('Sessions')
page_size = 50
page_number = st.number_input('Page Number', min_value=1, max_value=(len(conversations) // page_size) + 1, value=1)

start_idx = (page_number - 1) * page_size
end_idx = start_idx + page_size

st.table(conversations.iloc[start_idx:end_idx][['text', 'topic', 'sentiment']])
