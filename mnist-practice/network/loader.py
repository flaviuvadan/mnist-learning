""" MNIST loader """

import mnist


class Loader(object):
    """ Loader class to load training, validation, and test data from MNIST """

    def __init__(self):
        """ Init function """
        self.train_images = mnist.train_images()
        self.train_labels = mnist.train_labels()
        self.test_images = mnist.test_images()
        self.test_labels = mnist.test_labels()

    def test_data(self):
        """ Get paired test data """
        return zip(self.test_images, self.test_labels)

    def train_data(self):
        """ Get paired train data """
        return zip(self.train_images, self.train_labels)
