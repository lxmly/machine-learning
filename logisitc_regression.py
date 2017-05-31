# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression(object):
    """LR 
    """
    def __init__(self, learning_rate=0.05, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    #在X末尾添加bias项
    def add_bias(self, X):
        return np.insert(X, -1, 1, axis=1)

    def fit(self, X, y):
        X = self.add_bias(X)
        n_features = np.shape(X)[1]
        self.theta = np.zeros(n_features)

        for _ in range(self.n_iter):
            self.theta += self.learning_rate * X.T.dot((y - sigmoid(np.dot(X, self.theta))))

    def predict(self, X):
        X = self.add_bias(X)
        y_pred = sigmoid(X.dot(self.theta))
        #if <=0.5 then 0 else 1
        return np.round(y_pred).astype(int)