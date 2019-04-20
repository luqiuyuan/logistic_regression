import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

DIMENSION_INPUT = 49152
NUM_ITERATIONS = 100
m = 10 # Number of samples
alpha = 0.001 # Learning rate
SIZE_TESTING = 10 # Size of testing dataset

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Read inputs
def readInputs(path):
    X = np.zeros((DIMENSION_INPUT, 0))
    for filename in glob.glob(path):
        img = mpimg.imread(filename)
        X = np.hstack((X, img.reshape((DIMENSION_INPUT, 1))))
    # Normalize X so that it is in range [0, 1]
    X = X / 128 - 1
    return X

# Read ground truth
def readGroundTruth(path):
    Y = np.zeros((1, 0))
    file = open(path, "r")
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

######################
# Training phase
######################

# Read inputs and ground truth for training phase
X = readInputs("./dataset/training/*.jpg")
Y = readGroundTruth("./dataset/training/data.txt")

# (w, b) = initializeParametersZeros(DIMENSION_INPUT)
(w, b) = initializeParametersRandom(DIMENSION_INPUT)

Js = np.zeros((0))
for i in range(0, NUM_ITERATIONS):
    A = calculateForwardPropagatation(X, w, b)
    J = calculateCost(Y, A)[0, 0]
    Js = np.hstack((Js, np.array([J])))
    (dw, db) = calculateBackwardPropagation(X, Y, A)
    w = w - alpha * dw
    b = b - alpha * db

# Plot
fig = plt.figure()
plt.plot(np.arange(100), Js, label="Cost")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Training")
plt.show()

######################
# Testing phase
######################

# Read inputs and ground truth for testing phase
X = readInputs("./dataset/training/*.jpg")
Y = readGroundTruth("./dataset/testing/data.txt")

# Calculate the forward propagation for the testing dataset using trained parameters
A = calculateForwardPropagatation(X, w, b)
Y_hat = np.copy(A)
Y_hat[A>0.5] = 1
Y_hat[A<=0.5] = 0

# Calculate the precision
num_corrects = 0
for i in range(0, SIZE_TESTING):
    if Y[0, i] == Y_hat[0, i]:
        num_corrects += 1
precision = num_corrects / SIZE_TESTING
print("The precision of testing is " + str(precision*100) + "%.")
