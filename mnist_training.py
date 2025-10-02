import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

train = pd.read_csv("~/mnist_train.csv")
test = pd.read_csv("~/mnist_test.csv")

neurons = 10

X_train = train.iloc[:, 1:].values
y_train_raw = train.iloc[:, 0].values
X = np.array(X_train)
Y = np.array(y_train_raw)

X_test = test.iloc[:, 1:].values
Y_test_raw = test.iloc[:, 0].values

X = X / 255.0
X_test = X_test / 255.0

layer1weights = np.random.uniform(-0.1, 0.1, (784, neurons)).astype("float64")
layer2weights = np.random.uniform(-0.1, 0.1, (neurons, 10)).astype("float64")

layer1bias = np.zeros((1, neurons), dtype="float64")
layer2bias = np.zeros((1, 10), dtype="float64")


def one_hot(y, num_classes):
    N = y.shape[0]
    one_hot_y = np.zeros((N, num_classes))
    one_hot_y[np.arange(N), y] = 1
    return one_hot_y


def activation(yCap1):
    return np.maximum(0, yCap1)


def softmax(yCap2):
    exp_vals = np.exp(yCap2 - np.max(yCap2, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


size = Y.shape[0]
Y_one_hot = one_hot(Y, 10)


def backProp(learningRate):
    global layer1weights, layer1bias, layer2weights, layer2bias

    yCap1 = X @ layer1weights + layer1bias
    z1 = activation(yCap1)

    yCap2 = z1 @ layer2weights + layer2bias
    z2 = softmax(yCap2)

    errorMatrix = z2 - Y_one_hot

    delLBYdelW2 = (1 / size) * (z1.T @ errorMatrix)
    delLBYdelb2 = np.sum(errorMatrix, axis=0, keepdims=True) / size

    layer2weights -= learningRate * (delLBYdelW2)
    layer2bias -= learningRate * (delLBYdelb2)

    dZ1 = errorMatrix @ layer2weights.T

    dYCap1 = (yCap1 > 0).astype(float)
    errorMatrixLayer1 = dZ1 * dYCap1

    delLBYdelW1 = (1 / size) * (X.T @ errorMatrixLayer1)
    delLBYdelb1 = np.sum(errorMatrixLayer1, axis=0, keepdims=True) / size

    layer1weights -= learningRate * delLBYdelW1
    layer1bias -= learningRate * delLBYdelb1


def predict(X_data, W1, b1, W2, b2):
    yCap1 = X_data @ W1 + b1
    z1 = np.maximum(0, yCap1)
    yCap2 = z1 @ W2 + b2
    z2 = softmax(yCap2)

    predictions = np.argmax(z2, axis=1)
    return predictions, z2


def calculate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels) * 100

def save_model_parameters(w1, b1, w2, b2, filename="mnist_model_params.npz"):
    np.savez_compressed(
        filename,
        W1=w1,
        B1=b1,
        W2=w2,
        B2=b2
    )

itr = 1000
learningRate = 0.1

for i in range(1, itr + 1):
    backProp(learningRate)

    if (i % 200 == 0):
        train_predictions, _ = predict(X, layer1weights, layer1bias, layer2weights, layer2bias)
        train_accuracy = calculate_accuracy(train_predictions, y_train_raw)

        test_predictions, _ = predict(X_test, layer1weights, layer1bias, layer2weights, layer2bias)
        test_accuracy = calculate_accuracy(test_predictions, Y_test_raw)
        print('train accuracy = ', train_accuracy, '% test accuracy= ', test_accuracy, '%\n')

save_model_parameters(layer1weights, layer1bias, layer2weights, layer2bias)
print("done, data saved\n")
