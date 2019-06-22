""" Start learn class """


class StartLearn:
    """ Holds the configuration passed to learn.start() """

    def __init__(self,
                 net_1=False,
                 net_2=False,
                 net_3_fully_connected_softmax=False,
                 net_3_conv_connected_softmax=False,
                 net_3_relu=False,
                 net_3_dropout=False):
        """ Init function """
        self.net_1 = net_1
        self.net_2 = net_2
        self.net_3_fully_connected_softmax = net_3_fully_connected_softmax
        self.net_3_conv_connected_softmax = net_3_conv_connected_softmax
        self.net_3_relu = net_3_relu
        self.net_3_dropout = net_3_dropout
