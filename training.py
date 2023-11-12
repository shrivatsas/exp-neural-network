import numpy as np
from nn import forward_pass, backward_pass
import matplotlib.pyplot as plt
from setup import *

# Example dataset
input_data = np.array([[0,0], [0,1], [1,0], [1,1]])
actual_output = np.array([[0], [1], [1], [0]])

def train_neural_network(learning_rate, epochs):

    loss_history = []

    for epoch in range(epochs):
        hidden_layer_output, predicted_output = forward_pass(input_data)
        backward_pass(input_data, actual_output, hidden_layer_output, predicted_output, 
                      weights_hidden_to_output, weights_input_to_hidden, bias_output, bias_hidden, learning_rate)
        loss = np.mean(np.square(actual_output - predicted_output))
        loss_history.append(loss)

    return loss_history

learning_rates = [0.1, 0.01, 0.001]
epoch_settings = [1000, 5000, 10000]

plt.figure(figsize=(12,6))

for lr in learning_rates:
    for epochs in epoch_settings:
        loss_history = train_neural_network(lr, epochs)
        plt.plot(loss_history, label=f"LR: {lr}, Epochs: {epochs}")

plt.title("Neural Network training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()