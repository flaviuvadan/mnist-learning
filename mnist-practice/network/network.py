""" Network class file """

import numpy
import random


def sigmoid(x):
    """ Sigmoid function """
    return 1.0 / (1.0 + numpy.exp(-x))


class Network(object):
    """ Network class """

    def __init__(self, sizes):
        """
        Init function.
        :param sizes: [number of neurons in their respective layers]
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]  # random biases from a std normal distribution
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # random weights from std norm

    def feed_forward(self, a):
        """
        Feed forward step.
        :param a: network input
        :return: output of the network given a
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the network using a mini batch size of the training data
        :param training_data: [(input, output)]
        :param epochs: number of training steps
        :param mini_batch_size: training data batch size
        :param eta: learning rate
        :param test_data: the data to perform testing against
        """
        n_test = 0
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation to a single batch
        :param mini_batch: [(input, output)]
        :param eta: learning rate
        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(min(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b))]

    def evaluate(self, test_data):
        pass

    def backpropagation(self, x, y):
        """
        Return a tuple representing the gradient for the cost function.
        :param x: input
        :param y: desired output
        :return: (bias gradient, weight gradient)
        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]  # layer by layer activations
        zs = []  # layer by layer z vectors
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-1] = delta
            nabla_w[-1] = numpy.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)
