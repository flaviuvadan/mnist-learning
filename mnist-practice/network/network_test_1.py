""" Network test 1 """

import numpy
import random

from .cost import SimpleCost
from .network_blueprint import NetworkBlueprint


class NetworkTest1(NetworkBlueprint):
    """ Network Test 1 """

    def __init__(self, sizes, cost=SimpleCost):
        """
        Init function.
        :param sizes: [number of neurons in their respective layers]
        """
        super(NetworkTest1, self).__init__(sizes, cost)

    def init_biases(self):
        """ Randomly initialize weights from a standard normal """
        return [numpy.random.randn(y, 1) for y in self.sizes[1:]]

    def init_weights(self):
        """ Randomly initialize weights from a standard normal """
        return [numpy.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None, stdout=False):
        """
        Train the network using a mini batch size of the training data
        :param training_data: [(input, output)]
        :param epochs: number of training steps
        :param mini_batch_size: training data batch size
        :param eta: learning rate
        :param test_data: the data to perform testing against
        :param stdout: whether to print results during SGD
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
                evaluation = self.evaluate(test_data)
                curr_iteration_accuracy = (j, (evaluation / n_test) * 100)
                self.accuracy_per_epoch.append(curr_iteration_accuracy)
                if stdout:
                    print("Epoch {} : {} / {}".format(j, evaluation, n_test))
            else:
                if stdout:
                    print("Epoch {} complete".format(j + 1))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using back-propagation to a single batch
        :param mini_batch: [(input, output)]
        :param eta: learning rate
        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [new_bias + delta_new_bias for new_bias, delta_new_bias in zip(nabla_b, delta_nabla_b)]
            nabla_w = [new_weight + delta_new_weight for new_weight, delta_new_weight in zip(nabla_w, delta_nabla_w)]
        self.weights = [weight - (eta / len(mini_batch)) * new_weight for weight, new_weight in
                        zip(self.weights, nabla_w)]
        self.biases = [bias - (eta / len(mini_batch)) * new_bias for bias, new_bias in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        """
        Get the number of test inputs for which the networks outputs the correct result. Output is the index of
        whichever neuron in the final layer has the highest activation.
        :param test_data: data to perform testing against
        :return: int - number of correctly predicted inputs
        """
        test_results = [(numpy.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
