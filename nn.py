import numpy as np
from setup import *

# Set a random seed for reproducibility
np.random.seed(42)

# The sigmoid function as our simple activation function. The function maps any input to a value between 0 and 1, making it useful for binary classification.
def sigmoid(x: float):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: float):
    return sigmoid(x) * (1 - sigmoid(x))

# The forward pass involves calculating the output of the neural network for a given input.
def forward_pass(input_data):
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    return hidden_layer_output, predicted_output

# Backpropagation involves calculating the error in the output and then propagating this error backward through the network to update the weights.
def backward_pass(input_data, actual_output, hidden_layer_output, predicted_output,
                  weights_hidden_to_output, weights_input_to_hidden, bias_output, bias_hidden,
                  learning_rate):
    error = actual_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = np.dot(d_predicted_output, weights_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    weights_hidden_to_output += np.dot(hidden_layer_output.T, d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_to_hidden += np.dot(input_data.T, d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Simple training loop

if __name__ == '__main__':
    # Example dataset
    input_data = np.array([[0,0], [0,1], [1,0], [1,1]])
    actual_output = np.array([[0], [1], [1], [0]])

    # Training parameters
    learning_rate = 0.11
    epochs = 50000

    for epoch in range(epochs):
        hidden_layer_output, predicted_output = forward_pass(input_data)
        backward_pass(input_data, actual_output, hidden_layer_output, predicted_output,
                    weights_hidden_to_output, weights_input_to_hidden, bias_output, bias_hidden, learning_rate)

        if epoch % 1000 == 0:
            loss = np.mean(np.square(actual_output - predicted_output))
            print(f"Epoch {epoch} Loss: {loss}")