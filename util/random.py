from random import random

__author__ = 'Douglas'


def generate(minimum=-1, maximum=1):
    return (random() * (maximum - minimum)) + minimum