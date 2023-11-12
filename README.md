Tracing LLMs from the ground up

### 1. Writing a neural network

A neural network consists of 3 neuron layers
1. Input
2. Hidden
3. Output

During a training the network learns and adjusts **Weights** and **Biases**, They adjust the strength of the connections.

An **activation function** is a mathematical function applied to the output of each neuron. This introduces non-linearity into the model.

A **loss function** measures the difference between the model's predictions and actual data

Backpropagation and Optimization algorithms are used to update the model weights and biases based on the loss function.

Key building blocks
Neuron computation: Output = Activation(Weights * Input + Bias)
Layer-wise computation: Apply neuron computation across all neurons in a layer
Forward pass: Compute the output of the network by passing data across all layers in a sequence
Backward pass: Compute gradients (derivatives) of the loss function for each weight and bias
Updating parameters: Adjust the weights and bias using gradients to minimize the loss

#### Implementing in Python

Let's implement a simple neural network in Python. Our network will have:

    One input layer
    One hidden layer
    One output layer

We'll use Python's NumPy library for numerical operations. Here's a step-by-step guide:

1. Initialize Weights and Biases
2. Define the Activation Function and Its Derivative
3. Implement the Forward Pass Function
4. Implement the Backward Pass Function (Backpropagation)
5. Train the Network Using a Simple Dataset