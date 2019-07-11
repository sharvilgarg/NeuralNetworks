from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.10)
plt.figure(figsize=(10, 7))
plt.scatter(feature_set[:, 0], feature_set[:, 1], c=labels, cmap=plt.cm.winter)
plt.show()

labels = labels.reshape(100, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


np.random.seed(42)
weights = np.random.rand(2, 1)
lr = 0.5
bias = np.random.rand(1)

for epoch in range(200000):
    inputs = feature_set
    XW = np.dot(feature_set, weights) + bias
    z = sigmoid(XW)
    error_out = ((1 / 2) * (np.power((z - labels), 2)))
    print(error_out.sum())
    error = z - labels
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)
    z_delta = dcost_dpred * dpred_dz
    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)
    for num in z_delta:
        bias -= lr * num
