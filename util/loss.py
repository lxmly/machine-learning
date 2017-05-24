# -*- coding:utf-8 -*-

from __future__ import division

import numpy as np

class LeastSquaresLoss(object):

    def __init__(self):
        pass

    #一阶
    def grad(self, y, y_pred):
        return -1 * (y - y_pred)

    #二阶
    def hess(self, y, y_pred):
        return np.ones(np.shape(y))