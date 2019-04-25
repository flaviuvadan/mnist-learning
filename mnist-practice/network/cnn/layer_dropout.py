""" Dropout layer functions """

import randomgen
import theano
from theano import tensor
from theano.tensor import shared_randomstreams


def dropout_layer(layer, p_dropout):
    """
    Dropout some components of a layer according to p_dropout
    :param layer: given layer
    :param p_dropout: percent dropout
    :return: pruned layer
    """
    random_generator = randomgen.RandomGenerator()
    shared_random_number_generator = shared_randomstreams.RandomStreams(
        random_generator.randint(0, high=999999)
    )
    mask = shared_random_number_generator.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * tensor.cast(mask, theano.config.floatX)
