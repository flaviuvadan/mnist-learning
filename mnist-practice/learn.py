""" Learn uses the loader to feed the network and perform classification """

from network import cost, loader, network_test_1


def start():
    """ Start learning """
    load = loader.Loader()
    training_data, _, test_data = load.load_data()

    net = network_test_1.NetworkTest1([784, 30, 10])
    net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)


start()
