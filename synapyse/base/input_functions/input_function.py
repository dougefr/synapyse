from abc import ABCMeta, abstractmethod

__author__ = 'Douglas Eric Fonseca Rodrigues'


class InputFunction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate_output(self, input_connections):
        """
        :type input_connections: list[synapyse.base.connection.Connection]
        """
        pass