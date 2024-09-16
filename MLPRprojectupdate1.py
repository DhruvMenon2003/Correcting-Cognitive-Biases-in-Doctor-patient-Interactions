# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:33:52 2024

@author: lenovo
"""

#Step 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-4")
#The .encode() method converts a text string into a list of token integers.

encoding.encode("tiktoken is great!")
encoding.decode([83, 1609, 5963, 374, 2294, 0])
'tiktoken is great!'
#USe Chat GPT 4's advanced algorithms to solve my problem
nltk.download('punkt')
nltk.download('stopwords')
#Semi Supervised Learning
# Preprocess the data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(filtered_tokens)

df['Description'] = df['Description'].apply(preprocess_text)
df['Category'] = df['Category'].apply(preprocess_text)

# Split the data into training and test sets
X = df['Description']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_tfidf, y_train)
#Step8: Fit the model and make prediction.
# Make predictions on the test set
y_pred = clf.predict(X_test_tfidf)
#Step9: Calculate and print the accuracy. Also plot the confusion matrix shown below as an output reference.

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Balanced Accuracy:", balanced_accuracy)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=df['Category'].unique(),
            yticklabels=df['Category'].unique(), cbar=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

