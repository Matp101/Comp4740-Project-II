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
