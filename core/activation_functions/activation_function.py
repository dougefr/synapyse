from abc import ABCMeta, abstractmethod

__author__ = "Douglas Eric Fonseca Rodrigues"


class ActivationFunction:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.__x = 0.0
        self.y = self.calculate_output()
        self.dx = self.calculate_derivative()

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        """
        :type x: float
        """
        if self.__x != x:
            self.__x = x
            self.y = self.calculate_output()
            self.dx = self.calculate_derivative()

    def set_x(self, x):
        """
        :type x: float
        """
        self.x = x
        return self

    @abstractmethod
    def calculate_output(self):
        pass

    @abstractmethod
    def calculate_derivative(self):
        pass

    @abstractmethod
    def clone(self):
        pass
