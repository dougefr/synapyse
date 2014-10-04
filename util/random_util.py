from random import random

__author__ = 'Douglas Eric Fonseca Rodrigues'


def generate(minimum=-1, maximum=1):
    """
    :type minimum: float
    :type maximum: float
    """
    return (random() * (maximum - minimum)) + minimum