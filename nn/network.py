"""Neural Network class that ties everything together."""
import numpy as np
from .losses import MSE, CrossEntropy
from .optimizers import SGD


class NeuralNetwork:
    """A simple feedforward neural network."""
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)
        return self

    def compile(self, loss="mse", optimizer=None):
        if loss == "mse":
            self.loss_fn = MSE()
        elif loss == "cross_entropy":
            self.loss_fn = CrossEntropy()
        else:
            self.loss_fn = loss
        self.optimizer = optimizer or SGD(learning_rate=0.01)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        history = {"loss": []}
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch, y_batch = X[batch_idx], y[batch_idx]
                output = self.forward(X_batch)
                loss = self.loss_fn.forward(output, y_batch)
                epoch_loss += loss
                n_batches += 1
                self.backward()
                self.optimizer.update(self.layers)
            avg_loss = epoch_loss / n_batches
            history["loss"].append(avg_loss)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f}")
        return history

    def predict(self, X):
        return self.forward(X)
