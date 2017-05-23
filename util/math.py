import numpy as np


def major_vote(y):
    (values, counts) = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]