import json
import jsonpickle
from core.learning.training_set import TrainingSet
from impl.activation_functions.linear import Linear
from impl.activation_functions.sigmoid import Sigmoid
from impl.input_functions.weighted_sum import WeightedSum
from impl.multi_layer_perceptron import MultiLayerPerceptron

training_set = TrainingSet()

training_set.append([0.0, 0.0], [0.0])
training_set.append([0.0, 1.0], [1.0])
training_set.append([1.0, 0.0], [1.0])
training_set.append([1.0, 1.0], [0.0])

n = MultiLayerPerceptron(0.7, 0.9)

input_function = WeightedSum()
activation_function = Sigmoid()

n.add_layer(2, input_function, activation_function)
n.add_layer(3, input_function, activation_function)
n.add_layer(1, input_function, activation_function)

n.randomize_weights()

print(json.dumps(json.loads(jsonpickle.encode(n)), indent=4, sort_keys=True))

#n.learn(training_set)