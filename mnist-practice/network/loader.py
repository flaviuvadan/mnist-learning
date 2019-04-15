""" MNIST loader """

import gzip
import numpy
import pickle


class Loader:
    """ Loader class to load training, validation, and test data from MNIST """

    def __init__(self):
        """ Init function """
        pass

    def load_data(self):
        """ Load data """
        tr_d, va_d, te_d = self._load_data()
        training_inputs = [numpy.reshape(x, (784, 1)) for x in tr_d[0]]
        training_results = [self.vectorized_result(y) for y in tr_d[1]]
        training_data = zip(training_inputs, training_results)
        validation_inputs = [numpy.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = zip(validation_inputs, va_d[1])
        test_inputs = [numpy.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, te_d[1])
        return training_data, validation_data, test_data

    def _load_data(self):
        """ Read the content of mnist.pkl.gz (credit goes to Michael Nielsen) """
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            return pickle.load(f, encoding='latin1')

    def vectorized_result(self, j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network (credit goes to Michael Nielsen). """
        e = numpy.zeros((10, 1))
        e[j] = 1.0
        return e
