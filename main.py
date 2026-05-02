#!/usr/bin/env python3
"""Demo: Train a neural network on the XOR problem and the Iris dataset."""
import numpy as np
from nn import NeuralNetwork, Dense, Activation
from nn.losses import MSE, CrossEntropy
from nn.optimizers import Adam, SGD


def xor_demo():
    print("=" * 50)
    print("XOR Problem Demo")
    print("=" * 50)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = NeuralNetwork()
    model.add(Dense(2, 8))
    model.add(Activation("tanh"))
    model.add(Dense(8, 1))
    model.add(Activation("sigmoid"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.05))
    model.fit(X, y, epochs=1000, batch_size=4, verbose=True)

    predictions = model.predict(X)
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"  {X[i]} => {predictions[i][0]:.4f} (expected {y[i][0]})")


def iris_demo():
    print("\n" + "=" * 50)
    print("Iris Dataset Demo (synthetic)")
    print("=" * 50)
    np.random.seed(42)
    n_per_class = 50
    X_list, y_list = [], []
    centers = [[1, 1], [4, 4], [7, 1]]
    for i, center in enumerate(centers):
        X_list.append(np.random.randn(n_per_class, 2) * 0.8 + center)
        one_hot = np.zeros((n_per_class, 3))
        one_hot[:, i] = 1
        y_list.append(one_hot)
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    model = NeuralNetwork()
    model.add(Dense(2, 16))
    model.add(Activation("relu"))
    model.add(Dense(16, 8))
    model.add(Activation("relu"))
    model.add(Dense(8, 3))
    model.add(Activation("softmax"))
    model.compile(loss="cross_entropy", optimizer=Adam(learning_rate=0.01))
    model.fit(X, y, epochs=200, batch_size=16, verbose=True)

    preds = model.predict(X)
    accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))
    print(f"\nAccuracy: {accuracy*100:.1f}%")


if __name__ == "__main__":
    xor_demo()
    iris_demo()
