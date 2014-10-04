from abc import ABCMeta, abstractmethod

__author__ = 'Douglas Eric Fonseca Rodrigues'


class ActivationFunction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate_output(self, x):
        """
        :type x: float
        """
        pass

    @abstractmethod
    def calculate_derivative(self, x):
        """
        :type x: float
        """
        pass
