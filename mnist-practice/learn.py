""" Learn uses the loader to feed the network and perform classification """

from network import loader, network


class Parameters:
    """ Parameters class represents the parameters used as input for the network """
    input_layer = 784
    middle_layer = 30
    output_layer = 10

    def __init__(self, epochs, batch_size, learning_rate):
        """ Init function """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


def start():
    """ Start learning """
    parameters = Parameters(30, 10, 3.0)
    load = loader.Loader()
    net = network.Network([parameters.input_layer, parameters.middle_layer, parameters.output_layer])

    training_data, _, test_data = load.load_data()
    net.stochastic_gradient_descent(training_data, parameters.epochs, parameters.batch_size, parameters.learning_rate,
                                    test_data=test_data)


start()
