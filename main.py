import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

DIMENSION_INPUT = 49152
NUM_ITERATIONS = 100
m = 10 # Number of samples
alpha = 0.001 # Learning rate

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def readY():
    Y = np.zeros((1, 0))
    file = open("./dataset/training/data.txt", "r")
    for line in file:
        Y = np.hstack((Y, np.array([[int(line)]])))
    file.close()
    return Y

# Initialize parameters to all zeros
def initializeParametersZeros(dimension):
    return (np.zeros((dimension, 1)), 0)

def initializeParametersRandom(dimension):
    return (np.random.random((dimension, 1)) * 0.0001, 0)

def calculateForwardPropagatation(X, w, b):
    Z = w.T.dot(X) + b
    A = sigmoid(Z)
    return A

def calculateBackwardPropagation(X, Y, A):
    dw = X.dot((A - Y).T) / m
    db = (np.sum((A - Y), axis=1) / m)[0]
    return (dw, db)

def calculateCost(Y, A):
    return -(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))) / m

# Construct input matrix X
X = np.zeros((DIMENSION_INPUT, 0))
for filename in glob.glob("./dataset/training/*.jpg"):
    img = mpimg.imread(filename)
    X = np.hstack((X, img.reshape((DIMENSION_INPUT, 1))))
# Normalize X so that it is in range [0, 1]
X = X / 128 - 1

Y = readY()

# (w, b) = initializeParametersZeros(DIMENSION_INPUT)
(w, b) = initializeParametersRandom(DIMENSION_INPUT)

for i in range(0, NUM_ITERATIONS):
    A = calculateForwardPropagatation(X, w, b)
    J = calculateCost(Y, A)[0, 0]
    print(J)
    (dw, db) = calculateBackwardPropagation(X, Y, A)
    w = w - alpha * dw
    b = b - alpha * db
