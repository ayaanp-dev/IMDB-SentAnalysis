import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

tf.config.list_physical_devices('GPU')

x_train = np.load("x_train.npy")
y_train = np.load("y.npy")

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))

# print(x_train[0])

model = Sequential([
	LSTM(512, input_shape=(x_train.shape[1:]), return_sequences=True),
	Dropout(0.2),
	LSTM(512),
	Dropout(0.1),
	Dense(256, activation="relu"),
	Dropout(0.2),
	Dense(1, activation="sigmoid")
])

# print(model.summary())

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=250, batch_size=64)
model.save("model.h5")