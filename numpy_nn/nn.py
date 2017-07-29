from __future__ import print_function
import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))

    def train(self, inputs, outputs, iters):
        for i in xrange(iters):
            output = self.predict(inputs)
            error = outputs - output
            adjusment = np.dot(inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjusment

if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print('Random starting synaptic weights:')
    print(neural_network.synaptic_weights)

    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('New synaptic weights after training:')
    print(neural_network.synaptic_weights)

    print("Predicting:")
    print(neural_network.predict(np.array([1, 0, 0])))
