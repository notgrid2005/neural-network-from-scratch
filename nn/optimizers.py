"""Optimizer implementations."""
import numpy as np


class SGD:
    """Stochastic Gradient Descent with optional momentum."""
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            for j, (param, grad) in enumerate(layer.get_params()):
                key = (i, j)
                if key not in self.velocities:
                    self.velocities[key] = np.zeros_like(param)
                self.velocities[key] = self.momentum * self.velocities[key] - self.lr * grad
                param += self.velocities[key]


class Adam:
    """Adam optimizer."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            for j, (param, grad) in enumerate(layer.get_params()):
                key = (i, j)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
