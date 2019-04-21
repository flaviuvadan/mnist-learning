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

    @classmethod
    def vectorized_result(cls, j):
        """ Get a 10D unit vector with a 1.0 in the j'th position and zeroes elsewhere. Typical use case is to convert
        the activation layer of a net to a single output """
        e = numpy.zeros((10, 1))
        e[j] = 1.0
        return e
