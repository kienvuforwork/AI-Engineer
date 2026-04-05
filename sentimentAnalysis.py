import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from transformers import pipeline

data = pd.read_csv("book_reviews_sample.csv")
data["reviewText_clean"] = data["reviewText"].str.lower()

data["reviewText_clean"] = data.apply(lambda x : re.sub(r"([^\w\s])","", x["reviewText_clean"]), axis =1)

vader_sentiment = SentimentIntensityAnalyzer()
data['vader_sentiment_score'] = data["reviewText_clean"].apply(lambda review : vader_sentiment.polarity_scores(review)['compound'])

# create labels
bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']

data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)

data['vader_sentiment_label'].value_counts().plot.bar()
plt.show()

transformer_pipeline = pipeline("sentiment-analysis")
transformer_labels = []

for review in data['reviewText_clean'].values:
    sentiment_list = transformer_pipeline(review)
    sentiment_label = [sent['label'] for sent in sentiment_list]
    transformer_labels.append(sentiment_label)

data['transformer_sentiment_label'] = transformer_labels
data['transformer_sentiment_label'].value_counts().plot.bar()
plt.show()
