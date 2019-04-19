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
    def simple_cost(cls, output_activations, y):
        """
        Get the vector of partial derivatives for the output activations.
        :param output_activations: vector of layer output activations
        :param y: desired output
        :return: output - y
        """
        return output_activations - y
