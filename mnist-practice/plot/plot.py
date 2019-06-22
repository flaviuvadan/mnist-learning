""" Plot class file """

from matplotlib import pyplot, lines


class Plot:
    """ Plot class """

    def __init__(self, x_values, y_values, title, filename):
        """
        Initialize a plot class
        :param x_values: [x1, x2, ..., xn]
        :param y_values: [y1, y2, ..., yn]
        :param title: title of the plot
        :param filename: name of the file where the chart is saved
        """
        self.x_values = x_values
        self.y_values = y_values
        self.title = title
        self.filename = filename

    def plot(self):
        """ Plot the chart using the class' x and y values """
        fig, ax = pyplot.subplots()
        line = lines.Line2D(self.x_values, self.y_values)
        ax.add_line(line)
        ax.axis([0, len(self.x_values), 0, max(self.y_values)])
        pyplot.savefig(self.filename)
        fig.clear()
