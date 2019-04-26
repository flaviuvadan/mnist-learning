""" Miscellaneous function """


def size(data):
    """ Return the size of data """
    return data[0].get_value(borrow=True).shape[0]
