import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import joblib

os.environ["LOKY_MAX_CPU_COUNT"] = "2" 

dataset = load_dataset("LDJnr/Puffin")

conversations = pd.DataFrame(dataset['train'])

conversations['text'] = conversations['conversations'].apply(lambda x: " ".join([turn['value'] for turn in x]))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(conversations['text'])

kmeans = KMeans(n_clusters=10, random_state=42)
conversations['cluster'] = kmeans.fit_predict(X)

topics = {0: 'Programming Practices', 1: 'Customer Support'}  
conversations['topic'] = conversations['cluster'].map(topics).fillna('Misc')

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

if not os.path.exists('./data'):
    os.makedirs('./data')
conversations.to_csv('./data/processed_conversations.csv', index=False)

if not os.path.exists('./models'):
    os.makedirs('./models')
joblib.dump(kmeans, './models/kmeans_model.pkl')

