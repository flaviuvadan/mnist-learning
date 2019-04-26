""" Miscellaneous function """

import theano
from theano import tensor
import numpy

from ..loader import Loader


def size(data):
    """ Return the size of data """
    return data[0].get_value(borrow=True).shape[0]


def shared(data):
    """ Places data into shared variables for Theno """
    shared_x = theano.shared(numpy.asarray(data[0],
                                           dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(data[1],
                                           dtype=theano.config.floatX),
                             borrow=True)
    return shared_x, tensor.cast(shared_y, "int32")


def load_data_shared():
    """ Loads the Theano-shared MNIST data """
    loader = Loader()
    training_data, validation_data, test_data = loader.load_data()
    return [shared(training_data), shared(validation_data), shared(test_data)]
