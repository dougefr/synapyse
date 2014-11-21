import json

import jsonpickle


__author__ = "Douglas Eric Fonseca Rodrigues"


def print_json(o):
    """
    :type o: object
    """
    print(json.dumps(json.loads(jsonpickle.encode(o)), indent=4, sort_keys=True))