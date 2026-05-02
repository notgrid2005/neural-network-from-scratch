# 🧠 Neural Network From Scratch

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

A complete neural network library built **entirely from scratch** using only NumPy. No TensorFlow, no PyTorch — just pure math and Python.

## Features
- **Layers:** Dense (fully connected)
- **Activations:** ReLU, Sigmoid, Tanh, Softmax
- **Loss Functions:** MSE, Cross-Entropy
- **Optimizers:** SGD (with momentum), Adam
- **Mini-batch training** with shuffled indices
- **He weight initialization** for stable training

## Quick Start

```bash
pip install numpy
python main.py
```

## Usage

```python
from nn import NeuralNetwork, Dense, Activation
from nn.optimizers import Adam

model = NeuralNetwork()
model.add(Dense(2, 16))
model.add(Activation("relu"))
model.add(Dense(16, 1))
model.add(Activation("sigmoid"))

model.compile(loss="mse", optimizer=Adam(learning_rate=0.01))
model.fit(X_train, y_train, epochs=100, batch_size=32)
predictions = model.predict(X_test)
```

## Project Structure
```
neural-network-from-scratch/
├── nn/
│   ├── __init__.py       # Package exports
│   ├── layers.py         # Dense, Activation layers
│   ├── losses.py         # MSE, CrossEntropy
│   ├── network.py        # NeuralNetwork class
│   └── optimizers.py     # SGD, Adam
├── main.py               # XOR + Iris demos
├── requirements.txt
└── README.md
```

## License
MIT
