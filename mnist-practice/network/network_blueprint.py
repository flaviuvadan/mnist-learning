""" Network class blueprint that networks with different configurations can inherit from """

import numpy

from .cost import SimpleCost
from .functions import Functions


class NetworkBlueprint:
    """ Parent network class """

    def __init__(self, sizes, cost=SimpleCost):
        """ Init function """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = self.init_biases()
        self.weights = self.init_weights()
        self.cost = cost

        # follow list of class variables are for monitoring different metrics
        self.accuracy_per_epoch = []
        self.cost_per_epoch = []

    def init_biases(self):
        """ Initialize the biases of the network """
        raise NotImplementedError

    def init_weights(self):
        """ Initialize the weights of the network """
        raise NotImplementedError

    def feed_forward(self, a):
        """
        Feed forward step.
        :param a: network input
        :return: output of the network given a
        """
        for b, w in zip(self.biases, self.weights):
            a = Functions.sigmoid(numpy.dot(w, a) + b)
        return a

    def backpropagation(self, x, y):
        """
        Return a tuple representing the gradient for the cost function.
        :param x: input
        :param y: desired output
        :return: (bias gradient, weight gradient)
        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = Functions.sigmoid(z)
            activations.append(activation)

        # backward pass
        # this is the error of the last layer
        delta = self.cost.delta(zs[-1], activations[-1], y) * Functions.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        # going back through the net
        for l in range(2, self.num_layers):
            delta = numpy.dot(self.weights[-l + 1].transpose(), delta) * Functions.sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def get_accuracy_per_epoch(self):
        """ Get a list of tuples of accuracy vs. batch for the performed training steps """
        return [i[0] for i in self.accuracy_per_epoch], [i[1] for i in self.accuracy_per_epoch]

    def get_cost_per_epoch(self):
        """ Get a list of tuples of cost vs. batch for the performed training steps """
        return [i[0] for i in self.cost_per_epoch], [i[1] for i in self.cost_per_epoch]
