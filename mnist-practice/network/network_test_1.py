""" Network test 1 """

import numpy
import random

from .functions import Functions
from .network_blueprint import NetworkBlueprint


class NetworkTest1(NetworkBlueprint):
    """ Network Test 1 """

    def __init__(self, sizes):
        """
        Init function.
        :param sizes: [number of neurons in their respective layers]
        """
        super(NetworkTest1, self).__init__(sizes)

    def init_biases(self):
        """ Randomly initialize weights from a standard normal """
        return [numpy.random.randn(y, 1) for y in self.sizes[1:]]

    def init_weights(self):
        """ Randomly initialize weights from a standard normal """
        return [numpy.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the network using a mini batch size of the training data
        :param training_data: [(input, output)]
        :param epochs: number of training steps
        :param mini_batch_size: training data batch size
        :param eta: learning rate
        :param test_data: the data to perform testing against
        """
        training_data = list(training_data)
        n = len(training_data)

        n_test = 0
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def cost(self, x, y):
        """ Compute error of the output layer """
        return Functions.simple_cost(x, y)

    def evaluate(self, test_data):
        """
        Get the number of test inputs for which the networks outputs the correct result. Output is the index of
        whichever neuron in the final layer has the highest activation.
        :param test_data: data to perform testing against
        """
        test_results = [(numpy.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
