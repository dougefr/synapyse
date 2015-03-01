from synapyse.util import random_util

__author__ = 'Douglas Eric Fonseca Rodrigues'


class Connection:
    def __init__(self, origin, destination, weight=0.0):
        """
        :type origin: synapyse.base.neuron.Neuron
        :type destination: synapyse.base.neuron.Neuron
        :type weight: float
        """
        self.origin = origin
        self.destination = destination
        self.weight = weight

    def randomize_weight(self):
        self.weight = random_util.generate()