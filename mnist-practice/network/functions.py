""" Network activation and cost functions """

import numpy


class Functions:
    """ Functions class """

    @classmethod
    def sigmoid(cls, x):
        """ Sigmoid function """
        return 1.0 / (1.0 + numpy.exp(-x))

    @classmethod
    def sigmoid_prime(cls, x):
        """ Derivative of sigmoid function """
        return cls.sigmoid(x) * (1 - cls.sigmoid(x))
