#!/usr/bin/python

import random
import collections # you can use collections.Counter if you would like
import math

import numpy as np

from util import *

SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, interesting, great, plot, bored, not
    """
    # BEGIN_YOUR_ANSWER
    
    weights = {
        "so": 0,
        "interesting": 0,
        "great": 0,
        "plot": 0,
        "bored": 0,
        "not": 0
    }
    
    dataset = [
        ("so interesting", 1),
        ("great plot", 1),
        ("so bored", -1),
        ("not interesting", -1)
    ]
    
    step_size = 1
    
    def _gradient(_feature, y, _weights):
        return - y * int((dotProduct(_feature, _weights) * y) < 1)
    
    def _feature(x):
        list_of_words = x.split()
        dict_of_feature = {}
        for word in list_of_words:
            if word in dict_of_feature.keys():
                dict_of_feature[word] += 1
            else:
                dict_of_feature[word] = 1
        
        return dict_of_feature
    
    for data in dataset:
        x, y = data
        feature = _feature(x)
        
        gradient = _gradient(feature, y, weights)
        
        for f, v in feature.items():
            weights[f] -= step_size * gradient * v

    return weights
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER
    list_of_data = x.split()
    dict_of_word = collections.defaultdict(int)
    for word in list_of_data:
        if word not in dict_of_word.keys():
            dict_of_word[word] = 1
            continue
        
        dict_of_word[word] += 1
    
    return dict_of_word
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER
    
    def _gradient(feature, y, weights):
        return (-y) * sigmoid((-y) * dotProduct(feature, weights))

    for i in range(numIters):
        for x, y in trainExamples:
            feature = featureExtractor(x)
            
            # Initialize weights if not exist
            for f, v in feature.items():
                if f not in weights.keys():
                    weights[f] = 0

            # Compute gradient
            gradient = _gradient(feature, y, weights)
            
            # Update weights
            for f, v in feature.items():
                weights[f] -= eta * gradient * v
    
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER
    list_of_data = x.split()
    dict_of_ngram_word = {}
    
    for i in range(len(list_of_data) - n + 1):
        ngram_word = " ".join(list_of_data[i:i+n])
        if ngram_word not in dict_of_ngram_word.keys():
            dict_of_ngram_word[ngram_word] = 1
            continue
        
        dict_of_ngram_word[ngram_word] += 1
    
    return dict_of_ngram_word
    
    # END_YOUR_ANSWER

############################################################
# Problem 3: Multi-layer perceptron & Backpropagation
############################################################

class MLPBinaryClassifier:
    """
    A binary classifier with a 2-layer neural network
        input --(hidden layer)--> hidden --(output layer)--> output
    Each layer consists of an affine transformation and a sigmoid activation.
        layer(x) = sigmoid(x @ W + b)
    """
    def __init__(self):
        self.input_size = 2  # input feature dimension
        self.hidden_size = 16  # hidden layer dimension
        self.output_size = 1  # output dimension

        # Initialize the weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.init_weights()

    def init_weights(self):
        weights = np.load("initial_weights.npz")
        self.W1 = weights["W1"]
        self.W2 = weights["W2"]

    def forward(self, x):
        """
        Inputs
            x: input 2-dimensional feature (B, 2), B: batch size
        Outputs
            pred: predicted probability (0 to 1), (B,)
        """
        # BEGIN_YOUR_ANSWER
        def sigmoid(n):
            return 1 / (1 + np.exp(-n))
        
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        pred = self.a2
        
        return (pred.T).squeeze(0)
        # END_YOUR_ANSWER

    @staticmethod
    def loss(pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            loss: negative log likelihood loss, (B,)
        """
        # BEGIN_YOUR_ANSWER
        return - (target * np.log(pred) + (1 - target) * np.log(1 - pred))
        # END_YOUR_ANSWER

    def backward(self, pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            gradient: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
        """
        # BEGIN_YOUR_ANSWER
        loss = self.loss(pred, target)
        
        dL_da2 = - (target / self.a2 - (1 - target) / (1 - self.a2))
        da2_dz2 = self.a2 * (1 - self.a2)
        dz2_db2 = 1
        dz2_dW2 = self.a1
        dz2_da1 = self.W2
        da1_dz1 = self.a1 * (1 - self.a1)
        dz1_db1 = 1
        dz1_dW1 = self.x
        
        dL_db2 = dL_da2 * da2_dz2 * dz2_db2
        dL_dW2 = (dL_da2 * da2_dz2) @ dz2_dW2
        dL_db1 = (dL_da2 @ da2_dz2) @ dz2_da1.T @ da1_dz1.T * dz1_db1
        dL_dW1 = ((dL_da2 @ da2_dz2) @ dz2_da1.T @ da1_dz1.T) @ dz1_dW1
        
        gradient = {
            "W1": dL_dW1.T,
            "b1": dL_db1,
            "W2": dL_dW2.T,
            "b2": dL_db2
        }
        raise NotImplementedError()
        return gradient
        # END_YOUR_ANSWER
    
    def update(self, gradients, learning_rate):
        """
        A function to update the weights and biases using the gradients
        Inputs
            gradients: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
            learning_rate: step size for weight update
        Outputs
            None
        """
        # BEGIN_YOUR_ANSWER
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]
        
        # END_YOUR_ANSWER

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        """
        A training function to update the weights and biases using stochastic gradient descent
        Inputs
            X: input features, (N, 2), N: number of samples
            Y: true labels, (N,)
            epochs: number of epochs to train
            learning_rate: step size for weight update
        Outputs
            loss: the negative log likelihood loss of the last step
        """
        # BEGIN_YOUR_ANSWER
        for epoch in range(epochs):
            pred = self.forward(X)
            loss = self.loss(pred, Y)
            gradients = self.backward(pred, Y)
            self.update(gradients, learning_rate)
            
        return loss
        # END_YOUR_ANSWER

    def predict(self, x):
        return np.round(self.forward(x))