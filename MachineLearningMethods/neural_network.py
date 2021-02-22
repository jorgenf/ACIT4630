import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd


training_p = 0.7

data = pd.read_csv("../Data/new_dataset.csv")


values = data.values

training_data = values[0 : int(round(len(values) * training_p)), 1:]
testing_data = values[int(round(len(values) * training_p)):, 1:]
training_labels = values[0 : int(round(len(values) * training_p)), 0]
testing_labels = values[int(round(len(values) * training_p)):, 0]
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(53)))
model.add(tf.keras.layers.Dense(56, activation=tf.keras.activations.sigmoid))
model.add(tf.keras.layers.Dense(56, activation=tf.keras.activations.sigmoid))
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax))


model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics="accuracy")

model.fit(training_data, training_labels, batch_size=100, epochs=100)

model.evaluate(testing_data, testing_labels)