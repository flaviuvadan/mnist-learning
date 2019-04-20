""" Network test 2 """

from .network_blueprint import NetworkBlueprint


class NetworkTest2(NetworkBlueprint):
    """ Network Test 2 """

    def __init__(self, sizes):
        """ Init function """
        super(NetworkTest2, self).__init__(sizes)

    def init_biases(self):
        pass

    def init_weights(self):
        pass

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        pass

    def cost(self, x, y):
        pass
