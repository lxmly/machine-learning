# -*- coding:utf-8 -*-

from __future__ import division

import numpy as np
from decision_tree import ClassificationTree
from util.math import major_vote


class RandomForest(object):
    """随机森林
          相比于基于boosting方法的gbdt，随机森基于bagging方式，对样本集和特征集进行采样 
          整个处理过程还是相对简单的

       参数
       ----------
       n_estimators : int(default=100) 
                   树的数目
       max_depth : int(default=3) 
                   树的最大深度，它限制了树的结点数 
       min_samples_split : int(default=3)
                   树最小分割样本数 
       max_features : string, optional(default="sqrt")
                   If "sqrt", then max_features=sqrt(n_features)
                   If "log2", then max_features=log2(n_features)
    """

    def __init__(self,
                 n_estimators=100,
                 max_features="sqrt",
                 max_depth=5,
                 min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.y_shape = None

    def _sampling(self, X, y, sub_n_features, bootstrap=True):
        """样本和特征采样
           这里仿照scikit-learn的设计
           The sub-sample size is always the same as the original input sample size
           but the samples are drawn with replacement if bootstrap=True (default) 
        """
        #横向连接，方便切割
        X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)

        n_samples = X_y.shape[0]
        n_features = X_y.shape[1]
        sub_samples = n_samples

        if not bootstrap:
            sub_samples = sub_samples // 2

        #样本采样 纵向
        sample_index = np.random.choice(range(n_samples), size=sub_samples, replace=bootstrap)
        #特征采样 横向
        feature_index = np.random.choice(range(n_features), size=sub_n_features, replace=False)

        X_y = X_y[sample_index]

        return X_y[:, :-1], X_y[:,-1], feature_index

    def fit(self, X, y):
        n_features = X.shape[1]

        if self.max_features == "sqrt":
            max_n_features = int(np.sqrt(n_features))
        else:
            max_n_features = int(np.log2(n_features))

        for i in range(self.n_estimators):
            sub_X, sub_y, sub_feature_is = self._sampling(X, y, max_n_features)

            tree = ClassificationTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                feature_is=sub_feature_is
            )
            tree.fit(sub_X, sub_y)
            self.trees.append(tree)

    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            y_preds.append(self.trees[i].predict(X))
        y_preds = np.array(y_preds)

        y_pred = []
        for i in xrange(y_preds.shape[1]):
            y_pred.append(major_vote(y_preds[:,i]))

        return np.array(y_pred)