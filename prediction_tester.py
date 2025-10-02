import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def load_model_parameters(filename="mnist_model_params.npz"):
    data = np.load(filename)
    return data['W1'], data['B1'], data['W2'], data['B2']

def activation(yCap1):
    return np.maximum(0, yCap1)

def softmax(yCap2):
    exp_vals = np.exp(yCap2 - np.max(yCap2, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def predict(X_data, W1, b1, W2, b2):
    yCap1 = X_data @ W1 + b1
    z1 = activation(yCap1)
    yCap2 = z1 @ W2 + b2
    z2 = softmax(yCap2)
    predictions = np.argmax(z2, axis=1)
    return predictions, z2

test = pd.read_csv("~/mnist_test.csv")
X_test = test.iloc[:, 1:].values
Y_test_raw = test.iloc[:, 0].values
X_test = X_test / 255.0

try:
    W1, B1, W2, B2 = load_model_parameters()
except FileNotFoundError:
    print("error, file not found")
    exit()

image_index =  345
single_image_data = X_test[image_index, :]
single_image_reshaped = single_image_data.reshape(1, -1)

prediction_array, softmax_output = predict(single_image_reshaped, W1, B1, W2, B2)

predicted_digit = prediction_array[0]
true_digit = Y_test_raw[image_index]
confidence = softmax_output[0, predicted_digit] * 100

image_2d = single_image_data.reshape(28, 28)
plt.imshow(image_2d, cmap='gray')
print('predictedDigit = ', predicted_digit,' trueDigit = ', true_digit)
plt.axis('off')
plt.show()
