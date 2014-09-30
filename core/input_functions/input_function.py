from abc import ABCMeta, abstractmethod

__author__ = 'Douglas'


class InputFunction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate_output(self, input_connections):
        """
        :type input_connections: Iterable[Connection]
        """
        pass