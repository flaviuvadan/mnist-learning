""" Network class file """

import numpy
import random


def sigmoid(x):
    """ Sigmoid function """
    return 1.0 / (1.0 + numpy.exp(-x))


def sigmoid_prime(x):
    """ Derivative of sigmoid function """
    return sigmoid(x) * (1 - sigmoid(x))


def cost_derivative(output_activations, y):
    """
    Get the vector of partial derivatives for the output activations.
    :param output_activations: vector of layer output activations
    :param y: desired outcome
    :return:
    """
    return output_activations - y


class Network:
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
        training_data = list(training_data)
        n = len(training_data)

        n_test = 0
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

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
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        """
        Get the number of test inputs for which the networks outputs the correct result. Output is the index of
        whichever neuron in the final layer has the highest activation.
        :param test_data: data to perform testing against
        """
        test_results = [(numpy.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

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
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w
