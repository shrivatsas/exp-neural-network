import numpy as np

input_size = 2
hidden_size = 4
output_size = 1

# Initialize weights and biases
weights_input_to_hidden = np.random.randn(input_size, hidden_size)
weights_hidden_to_output = np.random.randn(hidden_size, output_size)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros(((1, output_size)))