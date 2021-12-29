import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Loading everything
data = pd.read_csv("D:/Datasets/IMDB Dataset.csv")
x_train = data["review"].tolist()
y_train = data["sentiment"].tolist()

# Making the labels binary
for i in range(0, 50000):
	if y_train[i] == "negative":
		y_train[i] = 0
	else:
		y_train[i] = 1

vectorizer = HashingVectorizer(n_features=20)
x_train = vectorizer.transform(x_train).toarray()
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

np.save("X.npy", x_train)
np.save("y.npy", y_train)