Tracing LLMs from the ground up

### 1. Writing a neural network

A neural network consists of 3 neuron layers
1. Input
2. Hidden
3. Output

During training the network learns and adjusts **Weights** and **Biases**, They adjust the strength of the connections.

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

#### Reflection

A slower learning rate with more iterations gave better output.

### 2. Sequence models

Sequence models are a type of model that is particularly adept at processing sequences of data. 

Temporal dependency, Sequence models are designed to handle data where the temporal order matters.
Recurrent Neural Networks, process sequences by maintaining a 'memory' (hidden state) of previous elements. This allows them to make predictions based on both the current input and what they've processed so far.
Long Short-term memory, A special kind of RNN designed to solve the problem of long-term dependencies.
Gated recurrent units, Similar to LSTMs, GRUs are a variation of RNNs that aim to solve the long-term dependency problem but with a simpler structure than LSTMs.
Attention mechanisms and Transformers, not RNNs, allow models to focus on different parts of the input sequence when producing each part of the output.
Sequence-to-Sequence Models, models are used for tasks where the input and output are both sequences

The backpropagation through time (BPTT) becomes complex for RNN compared to basic NNs
    Decide on a specific task for your RNN (e.g., sequence classification, language modeling).
    Prepare or create a dataset suitable for your chosen task.
    Implement the backward pass (BPTT) for training.
    Define a loss function and optimize the weights using gradient descent or a variant.

Select a task: Character-level language modeling. Train to predict the next character in a sequence given previous characters.

### 3. Attention mechanisms


### 4. Natural Language Processing

