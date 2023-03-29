"""
SKLearn on the Spooky Author Identification Challenge
"""

import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk as nl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read in the data
df = pd.read_csv('./res/train.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['author'], test_size=0.2, random_state=42)

# Create a CountVectorizer object
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test set
count_test = count_vectorizer.transform(X_test)

# Create a TfidfVectorizer object
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test set
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize a LogisticRegression object
logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(count_train, y_train)

# Predict the labels of the test set
y_pred = logreg.predict(count_test)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))
