""" Learn uses the loader to feed the network and perform classification """

from network import cnn, functions, loader, network_test_1, network_test_2, network_test_3


def message(net, extra):
    """
    Print the training warning message
    :param net: net index any([1, 2, 3])
    :param extra: extra information
    :return: formatted message
    """
    p = "=" * 25
    return "{} training network {}, message: {} {}".format(p, net, extra, p)


def start(net_1=False, net_2=False, net_3_fully_connected_softmax=False, net_3_conv_connected_softmax=False,
          run_net_3_with_relu=False, run_net_3_dropout=False):
    """ Start learning """
    load = loader.Loader()
    training_data, validation_data, test_data = load.load_data()

    if net_1:
        print(message(1, "simple net"))
        run_net_1(training_data, test_data)
    if net_2:
        print(message(2, "simple net"))
        run_net_2(training_data, test_data)
    if net_3_fully_connected_softmax:
        print(message(3, "CNN - layers = [fully connected, softmax]"))
        run_net_3_fully_connected_softmax()
    if net_3_conv_connected_softmax:
        print(message(3, "CNN - layers = [convolutional, connected, softmax]"))
        run_net_3_conv_fully_connected_softmax()
    if run_net_3_with_relu:
        print(message(3, "CNN - layers = [conv, conv, connected, softmax] with RELU"))
        run_net_3_conv_conv_fully_connected_softmax_with_relu()
    if run_net_3_dropout:
        print(message(3, "CNN - layers = [conv, conv, connected, connected, softmax] with RELU and dropout"))
        run_net_3_with_dropout()


def run_net_1(training_data, test_data):
    """ Runs net 1 """
    n_input, n_hidden, n_out = 784, 30, 10
    epochs, mini_batch_size, eta = 30, 10, 3.0
    net = network_test_1.NetworkTest1([n_input, n_hidden, n_out])
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, test_data=test_data)


def run_net_2(training_data, test_data):
    """ Runs net 2 """
    n_input, n_hidden, n_out = 784, 30, 10
    epochs, mini_batch_size, eta, lmbda = 60, 10, 0.5, 5
    net = network_test_2.NetworkTest2([n_input, n_hidden, n_out])
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, lmbda,
                                    evaluation_data=test_data,
                                    monitor_evaluation_accuracy=True,
                                    monitor_evaluation_cost=False,
                                    monitor_training_accuracy=True,
                                    monitor_training_cost=False)


def run_net_3_fully_connected_softmax():
    """ Runs net 3 with layers = [fully connected, softmax]"""
    fcn_n_inp, fcn_n_out = 784, 100
    sl_n_inp, sl_n_out = 100, 10
    epochs, mini_batch_size, eta = 60, 10, 0.1

    training_data, validation_data, test_data = cnn.load_data_shared()
    layers = [cnn.FullyConnectedLayer(fcn_n_inp, fcn_n_out), cnn.SoftmaxLayer(sl_n_inp, sl_n_out)]
    net = network_test_3.NetworkTest3(layers, mini_batch_size)
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, validation_data, test_data,
                                    print_mini_batch_iteration=False)


def run_net_3_conv_fully_connected_softmax():
    """ Runs net 3 with with layers = [convolutional, fully connected, softmax] """
    # 20 * 12 * 12 is the size of the pool layer, going into the fully connected layer
    fcn_n_inp, fcn_n_out = 20 * 12 * 12, 100
    sl_n_inp, sl_n_out = 100, 10
    epochs, mini_batch_size, eta = 60, 10, 0.1

    training_data, validation_data, test_data = cnn.load_data_shared()
    filter = (20, 1, 5, 5)
    image = (mini_batch_size, 1, 28, 28)
    # Conv layer default to pool of 2 x 2
    # Note: the convolution layer will perform the convolution and the pool layer operations using theano
    layers = [
        cnn.ConvPoolLayer(filter, image),
        cnn.FullyConnectedLayer(fcn_n_inp, fcn_n_out),  # fcn_n_out = sl_n_inp
        cnn.SoftmaxLayer(sl_n_inp, sl_n_out)
    ]
    net = network_test_3.NetworkTest3(layers, mini_batch_size)
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, validation_data, test_data,
                                    print_mini_batch_iteration=False)


def run_net_3_conv_conv_fully_connected_softmax():
    """ Runs net 3 with layers = [convolutional, convolutional, fully connected, softmax] """
    training_data, validation_data, test_data = cnn.load_data_shared()

    # there are 40 feature maps from the last pooling layer of 4 x 4 images
    fcn_n_inp, fcn_n_out = 40 * 4 * 4, 100
    sl_n_inp, sl_n_out = 100, 10
    epochs, mini_batch_size, eta = 60, 10, 0.1

    # original input image is a 1 dimensional 28 x 28 pixels
    conv_1_image = (mini_batch_size, 1, 28, 28)
    # this layer will have 20 feature maps, 1 input feature, and will use a 5 x 5 kernel
    conv_1_filter = (20, 1, 5, 5)

    # 20 12 x 12 feature maps coming from the previous pool layer
    conv_2_image = (mini_batch_size, 20, 12, 12)
    # 40 feature maps as input, 20 input features, and a 5 x 5 kernel
    conv_2_filter = (40, 20, 5, 5)

    layers = [
        cnn.ConvPoolLayer(conv_1_filter, conv_1_image),
        cnn.ConvPoolLayer(conv_2_filter, conv_2_image),
        cnn.FullyConnectedLayer(fcn_n_inp, fcn_n_out),
        cnn.SoftmaxLayer(sl_n_inp, sl_n_out),
    ]
    net = network_test_3.NetworkTest3(layers, mini_batch_size)
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, validation_data, test_data,
                                    print_mini_batch_iteration=False)


def run_net_3_conv_conv_fully_connected_softmax_with_relu():
    """ Runs net 3 with layers = [convolutional, convolutional, fully connected, softmax] and ReLU activation """
    functs = functions.Functions()
    training_data, validation_data, test_data = cnn.load_data_shared()

    fcn_n_inp, fcn_n_out = 40 * 4 * 4, 100
    sl_n_inp, sl_n_out = 100, 10
    epochs, mini_batch_size, eta, lmbda = 60, 10, 0.03, 0.1

    conv_1_image = (mini_batch_size, 1, 28, 28)
    conv_1_filter = (20, 1, 5, 5)

    conv_2_image = (mini_batch_size, 20, 12, 12)
    conv_2_filter = (40, 20, 5, 5)

    layers = [
        cnn.ConvPoolLayer(conv_1_filter, conv_1_image, activation_fn=functs.relu),
        cnn.ConvPoolLayer(conv_2_filter, conv_2_image, activation_fn=functs.relu),
        cnn.FullyConnectedLayer(fcn_n_inp, fcn_n_out, activation_fn=functs.relu),
        cnn.SoftmaxLayer(sl_n_inp, sl_n_out),
    ]
    net = network_test_3.NetworkTest3(layers, mini_batch_size)
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, validation_data, test_data,
                                    lmbda=lmbda,
                                    print_mini_batch_iteration=False)


def run_net_3_with_dropout():
    """ Runs net 3 with layers = [conv, conv, full, full, softmax] with ReLU activation and dropout """
    functs = functions.Functions()
    training_data, validation_data, test_data = cnn.load_data_shared()

    full_conn_1_input, full_conn_1_output = 40 * 4 * 4, 100
    full_conn_2_input, full_conn_2_output = 100, 100

    # drop half of the neurons
    p_dropout = 0.5

    sl_n_inp, sl_n_out = 100, 10
    epochs, mini_batch_size, eta, lmbda = 30, 10, 0.03, 0.1

    conv_1_image = (mini_batch_size, 1, 28, 28)
    conv_1_filter = (20, 1, 5, 5)

    conv_2_image = (mini_batch_size, 20, 12, 12)
    conv_2_filter = (40, 20, 5, 5)

    layers = [
        cnn.ConvPoolLayer(conv_1_filter, conv_1_image, activation_fn=functs.relu),
        cnn.ConvPoolLayer(conv_2_filter, conv_2_image, activation_fn=functs.relu),
        cnn.FullyConnectedLayer(full_conn_1_input, full_conn_1_output, activation_fn=functs.relu, p_dropout=p_dropout),
        cnn.FullyConnectedLayer(full_conn_2_input, full_conn_2_output, activation_fn=functs.relu, p_dropout=p_dropout),
        cnn.SoftmaxLayer(sl_n_inp, sl_n_out),
    ]
    net = network_test_3.NetworkTest3(layers, mini_batch_size)
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, validation_data, test_data,
                                    lmbda=lmbda,
                                    print_mini_batch_iteration=False)


start(net_1=False,
      net_2=False,
      net_3_fully_connected_softmax=False,
      net_3_conv_connected_softmax=False,
      run_net_3_with_relu=True,
      run_net_3_dropout=False)
