""" Learn uses the loader to feed the network and perform classification """

from network import loader, network_test_1, network_test_2


def start():
    """ Start learning """
    p = "=" * 50
    load = loader.Loader()
    training_data, validation_data, test_data = load.load_data()

    print("{} Training NetworkTest1 {}".format(p, p))
    net = network_test_1.NetworkTest1([784, 30, 10])
    net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)

    training_data, validation_data, test_data = load.load_data()
    print("{} Training NetworkTest2 {}".format(p, p))
    net2 = network_test_2.NetworkTest2([784, 30, 10])
    net2.stochastic_gradient_descent(training_data, 30, 10, 0.5, 5,
                                     evaluation_data=test_data,
                                     monitor_evaluation_accuracy=True,
                                     monitor_evaluation_cost=False,
                                     monitor_training_accuracy=True,
                                     monitor_training_cost=False)


start()
