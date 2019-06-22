""" Plot class file """

from matplotlib import pyplot, lines


class Plot:
    """ Plot class """

    def __init__(self, x_values, x_label, y_values, y_label, title, filename):
        """
        Initialize a plot class
        :param x_values: [x1, x2, ..., xn]
        :param x_label: string - label of the X axis
        :param y_values: [y1, y2, ..., yn]
        :param y_label: string - label of the Y axis
        :param title: title of the plot
        :param filename: name of the file where the chart is saved
        """
        self.x_values = x_values
        self.x_label = x_label
        self.y_values = y_values
        self.y_label = y_label
        self.title = title
        self.filename = filename

    def plot(self):
        """ Plot the chart using the class' x and y values """
        fig, ax = pyplot.subplots()
        line = lines.Line2D(self.x_values, self.y_values)
        ax.add_line(line)
        ax.set_title(self.title)
        ax.axis([0, len(self.x_values) + 1, 0, max(self.y_values) + 0.1])
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        pyplot.savefig(self.filename)
        fig.clear()
