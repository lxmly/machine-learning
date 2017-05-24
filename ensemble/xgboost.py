# -*- coding:utf-8 -*-

from __future__ import division

import numpy as np
from decision_tree import XGBoostRegressionTree
from util.loss import LeastSquaresLoss

class XGBoost(object):

    """梯度提升决策树
       
       参数
       ----------
       n_estimators : int(default=100) 
                   boosting stages执行的数目, Gradient boosting对于防止拟合比较健壮，
                   所以推荐选择一个较大的数目，会产生更好的表现
                   
       learning_rate : int(default=0.1) 
                   learning_rate 控制着每颗树的贡献，在n_estimators和learning_rate
                   间需要权衡一下(trade-off)
       
       max_depth : int(default=3) 
                   每颗树的最大深度，它限制了树的结点数 
                        
       min_samples_split : int(default=2)
                   树的最小分裂样本数 
       
       reg_lamda : float(default=0.1)   
                   正则系数
                     
    """
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.2,
                 max_depth=3,
                 min_samples_split=2,
                 reg_lambda=0.1):
        self.loss = LeastSquaresLoss()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.trees = []
        self.y_shape = None

    def fit(self, X, y):
        self.y_shape = np.shape(y)
        y_pred = np.zeros(self.y_shape)
        for i in range(self.n_estimators):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                loss=self.loss,
                reg_lambda=self.reg_lambda
            )

            #为了能够复用决策树框架，这里把y和y_pred进行了连接
            tree.fit(X, np.concatenate((np.expand_dims(y, axis=1), np.expand_dims(y_pred,axis=1)), axis=1))

            #Additive!累积!
            y_pred += self.learning_rate * tree.predict(X)

            #本轮stage完毕
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(self.y_shape)
        for i in range(self.n_estimators):
            y_pred += self.learning_rate * self.trees[i].predict(X)

        return y_pred
