from util import random

__author__ = 'Douglas'


class Connection:
    def __init__(self, origin, destination, weight=0.0):
        """
        :type origin: Neuron
        :type destination: Neuron
        :type weight: float
        """

        self.origin = origin
        """:type : Neuron"""

        self.destination = destination
        """:type : Neuron"""

        self.weight = weight
        """:type : float"""

    def randomize_weight(self):
        self.weight = random.generate()