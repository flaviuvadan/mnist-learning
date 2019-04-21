""" Network test 2 """

import numpy

from .network_blueprint import NetworkBlueprint


class NetworkTest2(NetworkBlueprint):
    """ Network Test 2 """

    def __init__(self, sizes, cost):
        """ Init function """
        super(NetworkTest2, self).__init__(sizes, cost)

    def init_biases(self):
        """ Initialize the biases based on a standard normal """
        return [numpy.random.randn(y, 1) for y in self.sizes[1:]]

    def init_weights(self):
        """ Initialize the weights based on a standard normal. The standard deviation is 1/sqrt(# neuron input
        connections) """
        return [numpy.random.randn(y, x) / numpy.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        pass

    def cost(self, a, y):
        """
        Compute error of the output layer
        :param a: activation
        :param y: desired output
        :return: float cost
        """
        return numpy.sum(numpy.nan_to_num(-y * numpy.log(a) - (1 - y) * numpy.log(1 - a)))

    def delta(self, z, a, y):
        """ Delta """
        return a - y
