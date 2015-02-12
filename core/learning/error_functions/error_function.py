from abc import ABCMeta, abstractmethod

__author__ = 'Douglas Eric Fonseca Rodrigues'


class ErrorFunction:
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def total_error(self):
        return 0

    @abstractmethod
    def add_error(self, output_error):
        """
        :type output_error: list[float]
        """
        pass

    @abstractmethod
    def reset(self):
        pass