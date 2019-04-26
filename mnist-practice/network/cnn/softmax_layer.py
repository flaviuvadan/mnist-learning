""" Softmax layer """

import theano
from theano import tensor
from theano.tensor import nnet
import numpy

from .conv_layer_blueprint import ConvLayerBlueprint
from .layer_dropout import dropout_layer


class SoftmaxLayer(ConvLayerBlueprint):
    """ Softmax layer """

    def __init__(self, n_in, n_out, p_dropout=0.0):
        """ Init function """
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.weights = None
        self.biases = None
        self.params = [self.weights, self.biases]
        self.inpt = None
        self.output = None
        self.y_out = None
        self.inpt_dropout = None
        self.output_dropout = None

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """ Set input """
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = nnet.softmax((1 - self.p_dropout) * tensor.dot(self.inpt, self.weights) + self.biases)
        self.y_out = tensor.argmax(self.output,
                                   axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = nnet.softmax(tensor.dot(self.inpt_dropout, self.weights) + self.biases)

    def init_weights(self):
        """ Initialize the weights """
        self.weights = theano.shared(
            numpy.zeros((self.n_in, self.n_out),
                        dtype=theano.config.floatX),
            name='w',
            borrow=True,
        )

    def init_biases(self):
        """ Initialize the biases """
        self.biases = theano.shared(
            numpy.zeros((self.n_out,),
                        dtype=theano.config.floatX),
            name='b',
            borrow=True,
        )

    def accuracy(self, y):
        """ Return the accuracy for a mini-batch """
        return tensor.mean(tensor.eq(y, self.y_out))

    def cost(self, net):
        """ Return the log-likelihood cost """
        return -tensor.mean(tensor.log(self.output_dropout)[tensor.arange(net.y.shape[0]), net.y])
