import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import math as m
import pandas as pd
import numpy as np

global train_X, train_Y, test_X, test_Y, W


def create_dataset(file_name, train_p, y_index, x_interval):
    global train_X, train_Y, test_X, test_Y, W
    file = pd.read_csv(file_name)

    values = tf.constant(file.values, dtype=tf.float32)
    tf.random.shuffle(values)



    train_stop = round(len(values) * train_p)
    test_start = train_stop
    samples = tf.shape(values)[0]
    train_shape = int(round(float(samples) * train_p))
    test_shape = int(round(float(samples) * (1 - train_p)))
    features = x_interval[1] - x_interval[0]
    train_ones = tf.cast(tf.ones((train_shape, 1)), tf.float32)
    test_ones = tf.cast(tf.ones((test_shape, 1)), tf.float32)

    train_X = tf.constant(values[0: train_stop, x_interval[0]: x_interval[1]], dtype=tf.float32, shape=(train_shape, features))
    train_X = tf.concat([train_X, train_ones], 1)
    train_Y = tf.constant(values[0:train_stop, y_index], dtype=tf.float32, shape=(train_shape,1))
    test_X = tf.constant(values[test_start:, x_interval[0]: x_interval[1]], dtype=tf.float32, shape=(test_shape, features))
    test_X = tf.concat([test_X, test_ones], 1)
    test_Y = tf.constant(values[test_start:, y_index], dtype=tf.float32, shape=(test_shape,1))

    # Initialize weights and bias
    W = tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=(len(train_X[0]), 1), dtype=tf.float32))


def hypothesis(X):
    global W
    Z = tf.matmul(X, W)
    Y_hat = tf.sigmoid(Z)
    return Y_hat


def gradient_descent(lr, epochs):
    global W
    len_W, n = W.get_shape()
    for it in range(epochs):
        Y_h = hypothesis(train_X)
        for w in range(len_W):
            diff = tf.subtract(Y_h, train_Y)
            X_w = tf.constant(train_X[:, w], shape=(len(train_X),1))
            loss = tf.multiply(diff, X_w)
            average_loss = tf.reduce_sum(loss) / len(train_X)
            cost = lr * average_loss
            W[w].assign(W[w]-cost)


create_dataset(file_name="breast_cancer_dataset.csv", train_p=0.7, y_index=1, x_interval=(2, 32))
#create_dataset(file_name="new_dataset.csv", train_p=0.7, y_index=0, x_interval=(1, 54))
gradient_descent(lr=0.01, epochs=1000)
Y_hat = hypothesis(test_X)
right = 0
wrong = 0
for y_h, y in zip(Y_hat, test_Y):
    if int(round(float(y_h))) == int(y):
        right += 1
    else:
        wrong += 1

print("Right: ", right)
print("Wrong: ", wrong)
