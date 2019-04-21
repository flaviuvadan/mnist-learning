""" Network test 2 """

import numpy
import random

from .cost import CrossEntropyCost
from .functions import Functions
from .network_blueprint import NetworkBlueprint


class NetworkTest2(NetworkBlueprint):
    """ Second network test """

    def __init__(self, sizes, cost=CrossEntropyCost):
        """ Init function """
        super(NetworkTest2, self).__init__(sizes, cost)

    def init_biases(self):
        """ Initialize the biases based on a standard normal """
        return [numpy.random.randn(y, 1) for y in self.sizes[1:]]

    def init_weights(self):
        """ Initialize the weights based on a standard normal. The standard deviation is 1/sqrt(# neuron input
        connections) """
        return [numpy.random.randn(y, x) / numpy.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, lmbda,
                                    evaluation_data=None,
                                    monitor_evaluation_cost=False,
                                    monitor_evaluation_accuracy=False,
                                    monitor_training_cost=False,
                                    monitor_training_accuracy=False):
        """
        Train the network using mini-batch stochastic gradient descent
        :param training_data: [(x, y)] training points
        :param epochs: number of training epochs
        :param mini_batch_size: size of training set/epoch
        :param eta: learning rate float
        :param lmbda: lambda (b/c lambda is a Python built-in) regularization parameter
        :param evaluation_data: test set to assess performance of the network
        :param monitor_evaluation_cost: boolean
        :param monitor_evaluation_accuracy: boolean
        :param monitor_training_cost: boolean
        :param monitor_training_accuracy: boolean
        :return: evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
        """
        n_data = None
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)

        evaluation_cost = []
        evaluation_accuracy = []
        training_cost = []
        training_accuracy = []
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch {} training complete".format(j))

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {} ".format(cost))

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}\n".format(accuracy, n_data))
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the network's weights and biases by applying SGD using backpropagation to a single batch
        :param training_data: [(x, y)] training points
        :param eta: learning rate float
        :param lmbda: lambda (b/c lambda is a Python built-in) regularization parameter
        :param n: size of training data set
        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1 - eta * (lmbda / n)) * weight - (eta / len(mini_batch)) * new_weight
                        for weight, new_weight in zip(self.weights, nabla_w)]
        self.biases = [bias - (eta / len(mini_batch)) * new_bias for bias, new_bias in zip(self.biases, nabla_b)]

    def accuracy(self, data, convert=False):
        """
        Get the number of inputs in data for which the network outputs the correct result. The network output is assumed
        to be the index of the neuron in the final layer with the highest activation.
        :param data: [(x, y)]
        :param convert: False if data is validation/test set, True if data is training set
        :return: number of correct predictions
        """
        if convert:
            # training data and validation/test data have different y structures
            results = [(numpy.argmax(self.feed_forward(x)), numpy.argmax(y)) for x, y in data]
        else:
            results = [(numpy.argmax(self.feed_forward(x)), y) for x, y in data]
        return sum(int(x == y) for x, y in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        Get the total cost for the set data.
        :param data: [(x, y)]
        :param lmbda: lambda (b/c lambda is a Python built-in) regularization parameter
        :param convert: False if data is validation/test set, True if data is training set
        :return: cost float
        """
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            if convert:
                y = Functions.vectorized_result(y)
            cost = cost + self.cost.fn(a, y) / len(data)
        return cost + 1 / 2 * (lmbda / len(data)) * sum(numpy.linalg.norm(w) ** 2 for w in self.weights)
