import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
import string
import nltk
from nltk.stem import WordNetLemmatizer
import threading
from sklearn.model_selection import train_test_split

# Loading everything
data = pd.read_csv("/home/ayaan/coding/IMDB-SentAnalysis/IMDB Dataset.csv")
x = data["review"].tolist()
y = data["sentiment"].tolist()

lemmatizer = WordNetLemmatizer()

# Making the labels binary
for i in range(0, 50000):
	if y[i] == "negative":
		y[i] = 0
	else:
		y[i] = 1

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
remove_punctuation(x)
print("noice 2")
lower_data(x)
print("noice 3")
lemmatize(x)
print("noice 4")
print(x[0])
print("noice 5")

vectorizer = HashingVectorizer(n_features=20)
x = vectorizer.transform(x).toarray()
x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)