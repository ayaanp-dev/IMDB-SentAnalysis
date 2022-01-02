import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
import string
import nltk
from nltk.stem import WordNetLemmatizer
import threading

# Loading everything
data = pd.read_csv("D:/Datasets/IMDB Dataset.csv")
x_train = data["review"].tolist()
y_train = data["sentiment"].tolist()

lemmatizer = WordNetLemmatizer()

# Making the labels binary
for i in range(0, 50000):
	if y_train[i] == "negative":
		y_train[i] = 0
	else:
		y_train[i] = 1

# Cleaning the data
def remove_punctuation(data):
	for index, text in enumerate(data):
		punctuation_free = "".join([i for i in text if i not in string.punctuation])
		data[index] = punctuation_free

def lower_data(data):
	for index, text in enumerate(data):
		data[index] = text.lower()

# def remove_stopwords(data):
# 	for i, comment in enumerate(data):
# 		data[i] = "".join(
#             filter(
#                 lambda word: word not in nltk.corpus.stopwords.words("english"), comment
#             )
#         )

def lemmatize(data):
	for index, text in enumerate(data):
		lemm_text = "".join([lemmatizer.lemmatize(word) for word in text])
		data[index] = lemm_text

print("noice 1")
remove_punctuation(x_train)
print("noice 2")
lower_data(x_train)
print("noice 3")
lemmatize(x_train)
print("noice 4")
print(x_train[0])
print("noice 5")

vectorizer = HashingVectorizer(n_features=20)
x_train = vectorizer.transform(x_train).toarray()
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

np.save("X.npy", x_train)
np.save("y.npy", y_train)