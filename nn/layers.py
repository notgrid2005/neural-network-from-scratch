"""Layer implementations for the neural network."""
import numpy as np


class Dense:
    """Fully connected layer."""
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.dweights = None
        self.dbias = None

    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.bias

    def backward(self, output_gradient):
        self.dweights = np.dot(self.input.T, output_gradient)
        self.dbias = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)

    def get_params(self):
        return [(self.weights, self.dweights), (self.bias, self.dbias)]


class Activation:
    """Activation function layer."""
    def __init__(self, activation="relu"):
        self.activation = activation
        self.input = None
        activations = {
            "relu": (self._relu, self._relu_prime),
            "sigmoid": (self._sigmoid, self._sigmoid_prime),
            "tanh": (self._tanh, self._tanh_prime),
            "softmax": (self._softmax, self._softmax_prime),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        self._forward_fn, self._backward_fn = activations[activation]

    def forward(self, input_data):
        self.input = input_data
        return self._forward_fn(input_data)

    def backward(self, output_gradient):
        return output_gradient * self._backward_fn(self.input)

    def get_params(self):
        return []

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_prime(x):
        return (x > 0).astype(float)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _tanh_prime(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def _softmax_prime(x):
        return np.ones_like(x)
