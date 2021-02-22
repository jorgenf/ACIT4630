import numpy as np
import random as rn


def iterative_gradient_decent(a_rate, it, data):
    training_data, testing_data, training_data_gt, testing_data_gt = data[0], data[1], data[2], data[3]
    W = np.array([rn.random() for n in range(len(training_data.values[0]) + 1)])
    for i in range(it):
        W = update_W(W, a_rate, training_data, training_data_gt)
    return W

def least_squared_error(W, training_data):
    J = 0
    m = len(training_data)
    for row in training_data.values:
        J += (np.power(W[0] + sum([w * x for w, x in zip(W[1:], row[:-1])]) - row[-1], 2) / (2 * m))
    return J


def update_W(W, a_rate, training_data, training_data_gt):
    m = len(training_data)
    for w in range(len(W)):
        dw = 0
        for training_row, gt_row in zip(training_data.values,training_data_gt.values):
            dw += (W[0] + sum([w * x for w, x in zip(W[1:], training_row[:-1])]) - gt_row[0]) * (training_row[w - 1] if w != 0 else 1)
        dw /= m
        W[w] -= a_rate * dw
    return W


def predict_data(W, testing_data):
    predicted_data = testing_data.values.copy()
    for data in predicted_data:
        data[0] = W[0] + sum(w * x for w, x in zip(W[1:], data[:-1]))
