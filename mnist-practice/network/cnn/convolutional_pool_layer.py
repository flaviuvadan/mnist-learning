""" Convolutional pool layer """

import numpy
import theano
from theano.tensor import nnet
from theano.tensor.signal import pool

from network.functions import Functions
from network.cnn.conv_layer_blueprint import ConvLayerBlueprint


class ConvPoolLayer(ConvLayerBlueprint):
    """ This layer is used to create a combination of convolutional and max-pooling layer. """

    def __init__(self, filter_shape, image_shape, pool_size=(2, 2), activation_fn=Functions.sigmoid):
        """
        Init function
        :param filter_shape: tuple(# filters, # input feature maps, filter height, filter width)
        :param image_shape: tuple(mini batch size, # input feature maps, image height, image width)
        :param poolsize: tuple(y, x) pooling sizes
        :param activation_fn: Functions.fn activation function
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.activation_fn = activation_fn
        self.weights = None
        self.biases = None
        self.inpt = None
        self.output = None
        self.output_dropout = None
        self.init_weights()
        self.init_biases()
        self.params = [self.weights, self.biases]

    def init_weights(self):
        """ Initialize the weights of the layer """
        n_out = (self.filter_shape[0] * numpy.prod(self.filter_shape[2:]) / numpy.prod(self.pool_size))
        self.weights = theano.shared(
            numpy.asarray(
                numpy.random.normal(loc=0, scale=numpy.sqrt(1 / n_out), size=self.filter_shape),
                dtype=theano.config.floatX,
            ),
            borrow=True,
        )

    def init_biases(self):
        """ Initialize the biases of the layer """
        self.biases = theano.shared(
            numpy.asarray(
                numpy.random.normal(loc=0, scale=1.0, size=(self.filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """
        Input setter
        :param inpt: image input from a layer
        :param inpt_dropout: input dropout
        :param mini_batch_size: mini batch size
        """
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = nnet.conv2d(self.inpt, self.weights,
                               filter_shape=self.filter_shape,
                               input_shape=self.image_shape)
        pooled_out = pool.pool_2d(conv_out,
                                  ws=self.pool_size,
                                  ignore_border=True)
        self.output = self.activation_fn(pooled_out + self.biases.dimshuffle('x', 0, 'x', 'x'))
        # no dropout in the conv layers
        self.output_dropout = self.output
