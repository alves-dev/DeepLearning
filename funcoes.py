import numpy as np
from numpy import signedinteger


def step(soma):
    if soma >= 1:
        return 1
    return 0

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def hyperbolicTangent(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def ReLU(soma):
    return max(0, soma)

def linear(soma):
    return soma

def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum()

print(sigmoid(2.1))
print(hyperbolicTangent(2.1))