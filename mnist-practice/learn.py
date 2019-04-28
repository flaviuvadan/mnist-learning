""" Learn uses the loader to feed the network and perform classification """

from network import cnn, loader, network_test_1, network_test_2, network_test_3


def start(net_1=False, net_2=False, net_3=False):
    """ Start learning """
    p = "=" * 50
    load = loader.Loader()
    training_data, validation_data, test_data = load.load_data()

    if net_1:
        print("{} Training NetworkTest1 {}".format(p, p))
        run_net_1(training_data, test_data)
    if net_2:
        print("{} Training NetworkTest2 {}".format(p, p))
        run_net_2(training_data, test_data)
    if net_3:
        print("{} TrainingNetwork3 {}".format(p, p))
        run_net_3()


def run_net_1(training_data, test_data):
    """ Runs net 1 """
    n_input, n_hidden, n_out = 784, 30, 10
    epochs, mini_batch_size, eta = 30, 10, 3.0
    net = network_test_1.NetworkTest1([n_input, n_hidden, n_out])
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, test_data=test_data)


def run_net_2(training_data, test_data):
    """ Runs net 2 """
    n_input, n_hidden, n_out = 784, 30, 10
    epochs, mini_batch_size, eta, lmbda = 30, 10, 0.5, 5
    net2 = network_test_2.NetworkTest2([n_input, n_hidden, n_out])
    net2.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, lmbda,
                                     evaluation_data=test_data,
                                     monitor_evaluation_accuracy=True,
                                     monitor_evaluation_cost=False,
                                     monitor_training_accuracy=True,
                                     monitor_training_cost=False)


def run_net_3():
    """ Runs net 3 """
    fcn_n_inp, fcn_n_out = 784, 100
    sl_n_inp, sl_n_out = 100, 10
    epochs, mini_batch_size, eta = 60, 10, 0.1

    training_data, validation_data, test_data = cnn.load_data_shared()
    layers = [cnn.FullyConnectedLayer(fcn_n_inp, fcn_n_out), cnn.SoftmaxLayer(sl_n_inp, sl_n_out)]
    net3 = network_test_3.NetworkTest3(layers, mini_batch_size)
    net3.stochastic_gradient_descent(training_data, 60, mini_batch_size, eta, validation_data, test_data)


start(net_1=False, net_2=False, net_3=True)
