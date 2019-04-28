""" Fully connected layer """

import numpy
import theano
from theano import tensor

from .conv_layer_blueprint import ConvLayerBlueprint
from .layer_dropout import dropout_layer
from ..functions import Functions


class FullyConnectedLayer(ConvLayerBlueprint):
    """ Fully connected layer """

    def __init__(self, n_in, n_out, activation_fn=Functions.sigmoid, p_dropout=0.0):
        """ Init function """
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.weights = None
        self.biases = None
        self.inpt = None
        self.inpt_dropout = None
        self.output = None
        self.output_dropout = None
        self.y_out = None
        self.init_weights()
        self.init_biases()
        self.params = [self.weights, self.biases]

    def init_weights(self):
        """ Initialize the weights """
        self.weights = theano.shared(
            numpy.asarray(
                numpy.random.normal(
                    loc=0.0,
                    scale=numpy.sqrt(1.0 / self.n_out),
                    size=(self.n_in, self.n_out)),
                dtype=theano.config.floatX,
            ),
            name='w',
            borrow=True,
        )

    def init_biases(self):
        """ Initialize the biases """
        self.biases = theano.shared(
            numpy.asarray(
                numpy.random.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(self.n_out,)
                ),
                dtype=theano.config.floatX,
            ),
            name='b',
            borrow=True,
        )

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """ Set input """
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1 - self.p_dropout) * tensor.dot(self.inpt, self.weights) + self.biases)
        self.y_out = tensor.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(tensor.dot(self.inpt_dropout, self.weights) + self.biases)

    def accuracy(self, y):
        """ Get accuracy for a given mini-batch """
        return tensor.mean(tensor.eq(y, self.y_out))
