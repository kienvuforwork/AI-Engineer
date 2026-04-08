from itertools import accumulate

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv("data/Restaurant_Reviews.tsv", delimiter="\t", quoting = 3)

corpus = []
stop_words = set(stopwords.words('english'))
stop_words.remove("not")
ps = PorterStemmer()
for i in range(0, 1000):
    review = re.sub(r"\W", " ", data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    corpus.append(review)

cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
y = data["Liked"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(np.concatenate(
    (y_pred.reshape(-1, 1), y_test.values.reshape(-1, 1)),
    axis=1
))
