""" Network class blueprint that networks with different configurations can inherit from """

import numpy

from .functions import Functions


class NetworkBlueprint:
    """ Parent network class """

    def __init__(self, sizes):
        """ Init function """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = self.init_biases()
        self.weights = self.init_weights()

    def init_biases(self):
        """ Initialize the biases and weights of the network """
        raise NotImplementedError

    def init_weights(self):
        """ Initialize the biases and weights of the network """
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

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the network using a mini batch size of the training data
        :param training_data: [(input, output)]
        :param epochs: number of training steps
        :param mini_batch_size: training data batch size
        :param eta: learning rate
        :param test_data: the data to perform testing against
        """
        raise NotImplementedError

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

    def cost(self, x, y):
        """ Cost function """
        raise NotImplementedError

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
        delta = self.cost(activations[-1], y) * Functions.sigmoid_prime(zs[-1])
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
            delta = numpy.dot(self.weights[-l + 1].transpose(), delta) * Functions.sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w
