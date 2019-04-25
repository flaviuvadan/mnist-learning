""" Fully connected layer """

from .conv_layer_blueprint import ConvLayerBlueprint


class FullyConnectedLayer(ConvLayerBlueprint):
    """ Fully connected layer """

    def __init__(self):
        """ Init function """
        pass

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """ Set input """
        pass

    def accuracy(self, y):
        pass
