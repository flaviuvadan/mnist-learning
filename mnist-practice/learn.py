""" Learn uses the loader to feed the network and perform classification """

from network import loader, network


class Parameters:
    """ Parameters class represents the parameters used as input for the network """
    input_layer = 784
    middle_layer = 30
    output_layer = 10

    epochs = 30
    mini_batch_size = 10
    learning_rate = 3.0


def start():
    parameters = Parameters()
    load = loader.Loader()
    net = network.Network([parameters.input_layer, parameters.middle_layer, parameters.output_layer])

    training_data, _, test_data = load.load_data()
    net.stochastic_gradient_descent(training_data, parameters.epochs, parameters.mini_batch_size,
                                    parameters.learning_rate,
                                    test_data=test_data)


start()
