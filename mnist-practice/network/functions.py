""" Network activation and cost functions """

import json
import numpy
import sys
from theano.tensor import nnet


class Functions:
    """ Functions class """

    @classmethod
    def relu(cls, x):
        """ Rectified linear unit function """
        return nnet.relu(x)

    @classmethod
    def sigmoid(cls, x):
        """ Sigmoid function """
        return 1.0 / (1.0 + numpy.exp(-x))

    @classmethod
    def sigmoid_prime(cls, x):
        """ Derivative of sigmoid function """
        return cls.sigmoid(x) * (1 - cls.sigmoid(x))

    @classmethod
    def vectorized_result(cls, j):
        """ Get a 10D unit vector with a 1.0 in the j'th position and zeroes elsewhere. Typical use case is to convert
        the activation layer of a net to a single output """
        e = numpy.zeros((10, 1))
        e[j] = 1.0
        return e

    @classmethod
    def save(cls, net, filename):
        """ Save the net config to filename """
        data = {
            "sizes": net.sizes,
            "weights": [w.tolist() for w in net.weights],
            "biases": [b.tolist() for b in net.biases],
            "cost": str(net.cost.__name__),
            "net": str(net.__name__)
        }

        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename):
        """ Load the network config from filename """
        from .nets import NETS
        with open(filename, "r") as f:
            data = json.loads(f)
            cost = getattr(sys.modules[__name__], data["cost"])
            net_name = getattr(sys.modules[__name__], data["net"])
            net_type = NETS.get(net_name)
            net = net_type(data["sizes"], cost=cost)
            net.weights = [numpy.array(w) for w in data["weights"]]
            net.biases = [numpy.array(b) for b in data["biases"]]
            return net
