import numpy as np

input_size = 
hidden_size = 10
output_size =

weights_input_to_hidden = np.random.randn(hidden_size, input_size) * 0.01
weights_hidden_to_hidden = np.random.randn(hidden_size, hidden_size) * 0.01
weights_hidden_to_output = np.random.randn(hidden_size, output_size) * 0.01

bias_hidden = np.zeros((hidden_size, 1))
bias_output = np.zeros((output_size, 1))

def rnn_step_forward(input_data, hidden_previous, weights_input_to_hidden, weights_hidden_to_hidden, weights_hidden_to_output, bias_hidden, bias_output):
    hidden_next = np.tanh(np.dot(weights_input_to_hidden, input_data) + np.dot(weights_hidden_to_hidden, hidden_previous))
    predicted_output = np.dot(weights_hidden_to_output, hidden_next)

    return hidden_next, predicted_output

# Sample dataset
text = "To be, or not to be, that is the question."
