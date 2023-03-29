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
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['author'], test_size=0.2, random_state=42)

# CountVectorizer
# A count_vectorizer object is used to transform a dataset of text into a matrix of token counts
count_vectorizer = CountVectorizer(stop_words='english')
# Learn the vocabulary dictionary and return document-term matrix.
# This is equivalent to fit followed by transform, but more efficiently implemented.
count_train = count_vectorizer.fit_transform(X_train)
# then we just transform the test data, using the same vocabulary learned from the training data
count_test = count_vectorizer.transform(X_test)
# use the logistic regression model to fit the data
logreg = LogisticRegression()
logreg.fit(count_train, y_train)
# predict the labels for the test data and print the accuracy
y_pred = logreg.predict(count_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))

# TFIDF
# Convert a collection of raw documents to a matrix of TF-IDF features.
# Equivalent to CountVectorizer followed by TfidfTransformer.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
tflogreg = LogisticRegression()
tflogreg.fit(tfidf_train, y_train)
tfy_pred = tflogreg.predict(tfidf_test)
tfaccuracy = accuracy_score(y_test, tfy_pred)
print("TFIDF Accuracy: {}".format(tfaccuracy))

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# Naive Bayes
nb = MultinomialNB()
nb.fit(tfidf_train, y_train)
nby_pred = nb.predict(tfidf_test)
nbaccuracy = accuracy_score(y_test, nby_pred)
print("Naive Bayes Accuracy: {}".format(nbaccuracy))

# SVM
svm = svm.SVC()
svm.fit(tfidf_train, y_train)
svmy_pred = svm.predict(tfidf_test)
svmaccuracy = accuracy_score(y_test, svmy_pred)
print("SVM Accuracy: {}".format(svmaccuracy))

# Random Forest
rf = RandomForestClassifier()
rf.fit(tfidf_train, y_train)
rfy_pred = rf.predict(tfidf_test)
rfaccuracy = accuracy_score(y_test, rfy_pred)
print("Random Forest Accuracy: {}".format(rfaccuracy))

# # Read in the test data
# test_df = pd.read_csv('./res/test.csv')
# # Transform the test data using the same vocabulary learned from the training data
# test_tfidf = tfidf_vectorizer.transform(test_df['text'])
# # Predict the labels for the test data
# test_pred = rf.predict(test_tfidf)
# # Create a submission file
# submission = pd.DataFrame({'id': test_df['id'], 'author': test_pred})
# submission.to_csv('./res/submission.csv', index=False)
