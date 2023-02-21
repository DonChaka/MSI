import numpy as np
import torch
import math
import random


class WAN(object):  # Weight Agnostic Neural
    def __init__(self, init_shared_weight, input_shape: int, output_shape: int, n_hidden=50):
        self.input_size = input_shape
        self.output_size = output_shape

        self.aVec = np.random.randint(low=1, high=10, size=n_hidden + self.output_size)
        self.wKey = np.random.randint(low=0, high=n_hidden ** 2, size=n_hidden ** 2)
        self.weights = np.random.normal(0, 2, len(self.wKey))
        self.weight_bias = np.random.normal(0, 2, 1)

        nNodes = len(self.aVec)
        self.wVec = [0] * (nNodes * nNodes)
        for i in range(nNodes * nNodes):
            self.wVec[i] = 0

        self.set_weight(init_shared_weight, 0)

    def set_weight(self, weight, weight_bias):
        nValues = len(self.wKey)
        if type(weight_bias).__name__ not in ['int', 'long', 'float']:
            weight_bias = 0
        if type(weight).__name__ in ['int', 'long', 'float']:
            weights = [weight] * nValues
        else:
            weights = weight

        for i in range(nValues):
            k = self.wKey[i]
            self.wVec[k] = weights[i] + weight_bias

    def tune_weights(self):
        self.weights = np.random.normal(0, 2, len(self.wKey))
        self.weight_bias = np.random.normal(0, 2, 1)
        self.set_weight(self.weights, self.weight_bias)

    def get_action(self, old_state):
        nNodes = len(self.aVec)
        wMat = np.array(self.wVec).reshape((nNodes, nNodes))
        nodeAct = [0] * nNodes
        nodeAct[0] = 1
        for i in range(len(old_state)):
            nodeAct[i + 1] = old_state[i]
        for iNode in range(self.input_size + 1, nNodes):
            rawAct = np.dot(nodeAct, wMat[:, iNode:iNode + 1])  # TPJ
            rawAct = self.applyActSimple(self.aVec[iNode], rawAct.tolist()[0])
            nodeAct[iNode] = rawAct
        return nodeAct[-self.output_size:][0]

    def applyActSimple(self, actId, x):
        if actId == 1:
            return x
        elif actId == 2:
            return 0.0 if x <= 0.0 else 1.0  # unsigned step
        elif actId == 3:
            return math.sin(math.pi * x)
        elif actId == 4:
            return math.exp(-(x * x) / 2.0)  # gaussian with mean zero and unit variance 1
        elif actId == 5:
            return math.tanh(x)
        elif actId == 6:
            return (math.tanh(x / 2.0) + 1.0) / 2.0  # sigmoid
        elif actId == 7:
            return -x
        elif actId == 8:
            return abs(x)
        elif actId == 9:
            return max(x, 0)  # relu
        elif actId == 10:
            return math.cos(math.pi * x)
        else:
            print('unsupported actionvation type: ', actId)
            return None
