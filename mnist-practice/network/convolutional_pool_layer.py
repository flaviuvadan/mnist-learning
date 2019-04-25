""" Convolutional pool layer """

from .functions import Functions


class ConvPoolLayer:
    """ This layer is used to create a combination of convolutional and max-pooling layer. """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=Functions.sigmoid):
        """
        Init function
        :param filter_shape: tuple(# filters, # input feature maps, filter height, filter width)
        :param image_shape: tuple(mini batch size, # input feature maps, image height, image width)
        :param poolsize: tuple(y, x) pooling sizes
        :param activation_fn: Functions.fn activation function
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        self.weights = None
        self.biases = None
        self.init_weights()
        self.init_biases()

    def init_weights(self):
        pass

    def init_biases(self):
        pass
