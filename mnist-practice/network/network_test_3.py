""" Network test 3 - Theano-based """

import theano
from theano import tensor

from . import cnn


class NetworkTest3:

    def __init__(self, layers, mini_batch_size):
        """
        Init function
        :param layers: [layers] that describe the network architecture
        :param mini_batch_size: size to be used during training by SGD
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = tensor.matrix("x")
        self.y = tensor.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)

        self.output = self.layers[-1].ouput
        self.output_dropout = self.layers[-1].output_dropout

        self.test_mini_batch_predictions = None

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data,
                                    lmbda=0.0):
        """
        Perform stochastic gradient descent on the CNN
        :param training_data: [(x, y)]
        :param epochs: number of epoch to train for
        :param mini_batch_size: the size of the training input
        :param eta: learning rate
        :param validation_data: [(x, y)]
        :param test_data: [(x, y)]
        :param lmbda: regularization parameter (L2)
        """
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        training_batches = cnn.size(training_data) / mini_batch_size
        validation_batches = cnn.size(validation_data) / mini_batch_size
        test_batches = cnn.size(test_data) / mini_batch_size

        # no good way to write L2 in Python, go for el_two since l2 is challenging to understand
        el_two_norm_squared = sum([(layer.weights ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + ((1 / 2) / training_batches) * lmbda * el_two_norm_squared
        gradients = tensor.grad(cost, self.params)
        updates = [(param, param - eta * grad) for param, grad in zip(self.params, grads)]

        mini_batch_index = tensor.lscalar()
        start = mini_batch_index * self.mini_batch_size
        end = (mini_batch_index + 1) * self.mini_batch_size
        train_mini_batch = theano.function([mini_batch_index], cost,
                                           updates=updates,
                                           givens={
                                               self.x: training_x[start: end],
                                               self.y: training_y[start: end],
                                           })
        train_mini_batch_accuracy = None
        self.test_mini_batch_predictions = None
        top_validation_accuracy = 0.0
