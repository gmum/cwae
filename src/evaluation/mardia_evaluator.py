import numpy as np
import tensorflow as tf
from numpy.linalg import norm as norml2


def b_1_d(X, chunks: int = None):
    N = np.shape(X)[0]
    if chunks is None:
        return (1/N**2)*np.sum(np.matmul(X, X.T)**3)
    else:
        sum = 0
        Cs = np.array_split(X, chunks)
        for c2 in Cs:
            c2t = c2.T
            for c1 in Cs:
                sum += np.sum(np.matmul(c1, c2t)**3)

        return (1/N**2)*sum


def b_2_d(X):
    return np.mean(norml2(X, axis=1)**4)


def skewness_test(X, chunks: int = None):
    return b_1_d(X, chunks)


def kurtosis_test(X):
    return b_2_d(X)


def multinormal_test(X, chunks: int = None):
    skewness = skewness_test(X, chunks)
    kurtosis = kurtosis_test(X)
    return skewness, kurtosis


class MardiaEvaluator:

    def __init__(self, chunks: int = 2):
        self.__chunks = chunks

    def build(self):
        pass

    def evaluate(self, session: tf.Session, z):
        print('Computing skewness and kurtosis')
        s, k = multinormal_test(z, self.__chunks)
        return [('skewness', s), ('kurtosis', k)]
