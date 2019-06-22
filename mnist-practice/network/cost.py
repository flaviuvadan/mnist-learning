""" Holds cost functions' classes """

import numpy

from .functions import Functions


class Cost:
    """ Main Cost class """

    def __init__(self):
        """ Init function """
        pass

    @staticmethod
    def fn(a, y):
        """
        Get the cost
        :param a: activation
        :param y: desired output
        :return: cost
        """
        raise NotImplementedError

    @staticmethod
    def delta(z, a, y):
        """
        Get the delta of the cost function
        :param z: L-1 activation (L = layer)
        :param a: activation
        :param y: desired output
        :return: delta cost
        """
        raise NotImplementedError


class SimpleCost(Cost):
    """ Calculate a simple cost """

    def __init__(self):
        """ Init function """
        super(SimpleCost, self).__init__()

    @staticmethod
    def fn(a, y):
        """ Compute the error of an output layer """
        return a - y

    @staticmethod
    def delta(z, a, y):
        """ Get the delta of the cost function """
        return a - y


class QuadraticCost(Cost):
    """ Calculate the quadratic cost """

    def __init__(self):
        """ Init function """
        super(QuadraticCost, self).__init__()

    @staticmethod
    def fn(a, y):
        """ Compute error of an output layer """
        return 1 / 2 * numpy.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """ Get the delta of the cost function """
        return a - y


class CrossEntropyCost(Cost):
    """ Calculate the cross-entropy cost """

    def __init__(self):
        """ Init function """
        super(CrossEntropyCost, self).__init__()

    @staticmethod
    def fn(a, y):
        """ Compute error of an output layer """
        return numpy.sum(numpy.nan_to_num(-y * numpy.log(a) - (1 - y) * numpy.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """ Get the delta of the cost function """
        return a - y
