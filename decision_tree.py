# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np
from util.math import major_vote

class DecisionTree(object):
    """决策树 base
    """
    def __init__(self, min_samples_split=2, max_depth=10, feature_is=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self._compute_leaf_value = None
        self._compute_node_impurity=None
        self.feature_is = feature_is

    def fit(self, X, y):
        if self.feature_is is None:
            self.feature_is = np.array(range(np.shape(X)[1]))
        self.tree = self._build_tree(X, y, self.feature_is)

    def _build_tree(self, X, y, feature_is, depth=0):
        min_impurity = float('inf')
        best_mask = None
        best_feature = None
        if len(feature_is) >= 1 and np.shape(X)[0] >= self.min_samples_split \
                and depth <= self.max_depth:

            for feature_i in feature_is:
                # 第i个特征的值集合
                feature_i_vs = np.unique(X[:, feature_i])

                for feature_i_v in feature_i_vs:
                    # 基于特征二分(回归树为二叉树)
                    mask = X[:, feature_i] <= feature_i_v
                    y1 = y[mask]
                    y2 = y[~mask]
                    cur_impurity = self._compute_node_impurity(y, y1, y2)
                    if cur_impurity < min_impurity:
                        min_impurity = cur_impurity
                        best_mask = mask
                        best_feature = (feature_i, feature_i_v)

            # 如果分属于不同的类别
            if not np.all(best_mask):
                feature_is = feature_is[feature_is != best_feature[0]]
                left = self._build_tree(X[best_mask], y[best_mask], feature_is, depth + 1)
                right = self._build_tree(X[~best_mask], y[~best_mask], feature_is, depth + 1)
                return Node(feature_i=best_feature[0], feature_i_v=best_feature[1], left=left, right=right)

        return Node(leaf_value=self._compute_leaf_value(y))

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._predict(x, self.tree))
        return np.asarray(y_pred)

    def _predict(self, x, tree):
        if tree.leaf_value is not None:
            return tree.leaf_value

        if x[tree.feature_i] <= tree.feature_i_v:
            return self._predict(x, tree.left)

        return self._predict(x, tree.right)


class Node(object):
    """结点
       leaf_value ： 记录叶子结点值
       feature_i ：特征i
       feature_i_v ： 特征i的值
       left ： 左子树
       right ： 右子树
    """

    def __init__(self, leaf_value=None, feature_i=None, feature_i_v=None, left=None, right=None):
        self.leaf_value = leaf_value
        self.feature_i = feature_i
        self.feature_i_v = feature_i_v
        self.left = left
        self.right = right

class ClassificationTree(DecisionTree):

    def _entropy(self, y):
        n = len(y)
        entropy = 0.
        for label in np.unique(y):
            p = len(y[label == y]) / n
            entropy += -p * np.log2(p)
        return entropy

    #信息增益比 取反
    def _information_gain_ratio_reverse(self, y, y1, y2):
        n = len(y)
        entropy_y1 = self._entropy(y1)
        p_y1 = len(y1) / n
        entropy_y2 = self._entropy(y2)
        p_y2 = 1 - p_y1

        if p_y1 == 0 or p_y2 == 0:
            return 0
        information_gain = self._entropy(y) - (p_y1 * entropy_y1 + p_y2 * entropy_y2)
        ha = -p_y1 * np.log2(p_y1) + -p_y2 * np.log2(p_y2)
        information_gain_ratio = information_gain / ha

        return -information_gain_ratio

    def fit(self, X, y):
        self._compute_node_impurity = self._information_gain_ratio_reverse
        self._compute_leaf_value = major_vote
        super(ClassificationTree, self).fit(X, y)

class RegressionTree(DecisionTree):

    def _friedman_mse(self, yl, yr):
        # gbdt论文 公式35 (wl*wr / wl + wr) *(yl_mean - yr_mean)^2
        # 样本权重均默认为1
        wl = len(yl)
        wr = len(yr)
        yl_mean = np.mean(yl)
        yr_mean = np.mean(yr)

        return (wl * wr / (wl + wr)) * ((yl_mean - yr_mean) ** 2)

    def _mse(self, x):
        if len(x) == 0:
            return 0
        mean = np.mean(x)
        return np.mean((x - mean) ** 2)

    def _mean(self, y):
        return np.mean(y)

    def _node_mse(self, y, y1, y2):
        return self._mse(y1) + self._mse(y2)

    def fit(self, X, y):
        self._compute_node_impurity = self._node_mse
        self._compute_leaf_value = self._mean
        super(RegressionTree, self).fit(X, y)


class XGBoostRegressionTree(DecisionTree):
    """XGBoost回归树
    """

    def __init__(self, min_samples_split=2, max_depth=10, feature_is=None, reg_lambda=0.1, loss=None):
        DecisionTree.__init__(self, min_samples_split=min_samples_split, max_depth=max_depth, feature_is=feature_is)
        self.reg_lambda = reg_lambda
        self.loss = loss

    #xgboost paper中定义的增益
    def _compute_gain(self, yy, yy1, yy2):
        gain = 0.5 * (self._Gn_by_H(yy1, 2) + self._Gn_by_H(yy2, 2) - self._Gn_by_H(yy, 2))
        return -gain

    #计算G^2/H+lamda 或者 G/H+lamda
    def _Gn_by_H(self, yy, n=1):
        y, y_pred = self._separate(yy)
        G = np.sum(self.loss.grad(y, y_pred))
        H = np.sum(self.loss.hess(y, y_pred))
        return G ** n / (H + self.reg_lambda)

    #xgboost paper中叶子结点的权重计算
    def _compute_weight(self, yy):
        return -self._Gn_by_H(yy)

    def _separate(self, yy):
        mid = np.shape(yy)[1] // 2
        return yy[:, :mid], yy[:, mid:]

    def fit(self, X, yy):
        self._compute_node_impurity = self._compute_gain
        self._compute_leaf_value = self._compute_weight
        super(XGBoostRegressionTree, self).fit(X, yy)