# NN.py a simple implementation of a feedforward neural network
# This will be used to run a genetic algorithm to evolve the weights of the network

import numpy as np
from typing import Callable

class Neuron:
    def __init__(self, weights: np.ndarray, bias: float, activation: Callable[[float], float]) -> None:
        """
        Initializes the neuron with the given weights, bias, and activation function
        
        Parameters
        ----------
        weights: np.ndarray - the weights of the neuron
        bias: float - the bias of the neuron
        activation: Callable[[float], float] - the activation function of the neuron, takes in a float and returns a float.                    
        
        Returns
        -------
        None
        """
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> float:
        """
        Forward pass of the neuron, takes in inputs and returns the output of the neuron
        
        Parameters
        ----------
        inputs: np.ndarray - the inputs to the neuron
        
        Returns
        -------
        float - the output of the neuron
        """
                
        return self.activation(np.dot(self.weights, inputs) + self.bias)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the neuron
        
        Returns
        -------
        str - the string representation of the neuron
        """
        
        return "Neuron: " + str(self.weights) + " " + str(self.bias)

class Layer:    
    def __init__(self, neurons: list[Neuron], name: str = "") -> None:
        """
        Initializes the layer with the given neurons
        
        Parameters
        ----------
        neurons: list[Neuron] - the neurons of the layer
        name: str - the name of the layer (optional, defaults to "")
        
        Returns
        -------
        None
        """
        self.neurons = neurons
        self.name = name

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer, takes in inputs and returns the outputs of the layer
        
        Parameters
        ----------
        inputs: np.ndarray - the inputs to the layer
        
        Returns
        -------
        np.ndarray - the outputs of the layer
        """
        
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def __str__(self):
        if self.name == "":
            return f"Unnamed Layer with {len(self.neurons)} neurons\n" + "\n".join([str(neuron) for neuron in self.neurons])
        else:
            return f"{self.name} with {len(self.neurons)} neurons\n" + "\n".join([str(neuron) for neuron in self.neurons])
    
class NeuralNetwork:
    def __init__(self, layers: list[Layer]) -> None:
        """
        Initializes the neural network with the given layers
        
        Parameters
        ----------
        layers: list[Layer] - the layers of the neural network
            
        Returns
        -------
        None
        """
        self.layers = layers

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the neural network
        
        Parameters
        ----------
        inputs: np.ndarray - the inputs to the neural network
        
        Returns
        -------
        np.ndarray - the outputs of the neural network
        """
                        
        for layer in self.layers:
            outputs = layer.forward(inputs)
            inputs = outputs  # Set the inputs for the next layer to the outputs of the current layer            
        
        return outputs
    
    def crossover(self, other: "NeuralNetwork") -> "NeuralNetwork":
        """
        Crossover operation between two neural networks
        
        Parameters
        ----------
        other: NeuralNetwork - the other neural network to crossover with
        
        Returns
        -------
        NeuralNetwork - the offspring of the two neural networks
        """
        
        new_layers = []
        for layer1, layer2 in zip(self.layers, other.layers):
            new_neurons = []
            for neuron1, neuron2 in zip(layer1.neurons, layer2.neurons):
                new_weights = []
                for w1, w2 in zip(neuron1.weights, neuron2.weights):
                    new_weights.append(w1 if np.random.rand() > 0.5 else w2)
                new_bias = neuron1.bias if np.random.rand() > 0.5 else neuron2.bias
                new_neurons.append(Neuron(np.array(new_weights), new_bias, neuron1.activation))
            new_layers.append(Layer(new_neurons))
        
        return NeuralNetwork(new_layers)
    
    def mutate(self, mutation_rate: float) -> None:
        """
        Mutates the neural network with the given mutation rate
        
        Parameters
        ----------
        mutation_rate: float - the probability of a weight being mutated
        
        Returns
        -------
        None
        """
        
        for layer in self.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    if np.random.rand() < mutation_rate:
                        neuron.weights[i] = np.random.randn()
                if np.random.rand() < mutation_rate:
                    neuron.bias = np.random.randn()                    

    def __str__(self) -> str:
        """
        Returns a string representation of the neural network
        
        Returns
        -------
        str - the string representation of the neural network
        """
        
        out = f"A neural network with {len(self.layers)} layers.\n"
        out += "\n".join([str(layer) for layer in self.layers])
        
        return out
    
def sigmoid(x: float) -> float:
    """
    Sigmoid activation function
    
    Parameters
    ----------
    x: float - the input to the activation function
    
    Returns
    -------
    float - the output of the activation function
    """
    
    return 1 / (1 + np.exp(-x))

def relu(x: float) -> float:
    """
    ReLU activation function
    
    Parameters
    ----------
    x: float - the input to the activation function
    
    Returns
    -------
    np.float - the output of the activation function
    """
    
    return max(0, x)

def createNeuralNetwork(n_inputs:int, n_hidden:list[int], n_outputs:int, activation:Callable[[float], float], connectivity:float) -> NeuralNetwork:
    """
    Creates a neural network with the given number of inputs, hidden layers, outputs, and activation function, with randomly initialised weights and biases
    
    Parameters
    ----------
    n_inputs: int - the number of inputs to the neural network
    n_hidden: list[int] - the number of neurons in each hidden layer
    n_outputs: int - the number of outputs of the neural network
    activation: Callable[[float], float] - the activation function of the neurons.functions, one for each layer.
    connectivity: float - the probability of a connection between two neurons, in the range [0.05, 1]. A value of 1 means all neurons are connected to all other neurons. We avoid values less than 0.05 to ensure the network is not too sparse.
    
    Returns
    -------
    NeuralNetwork - the neural network
    """
    
    layers = []
    prev_n = n_inputs
    
    for layer_num, n in enumerate(n_hidden):
        neurons = []
        for i in range(n):            
            weights = np.random.randn(prev_n)
            bias = np.random.randn()
            neurons.append(Neuron(weights, bias, activation))
        layers.append(Layer(neurons, f"Hidden Layer {layer_num + 1}"))
        prev_n = n
    # Output layer
    out_neurons = []
    for i in range(n_outputs):
        weights = np.random.randn(prev_n)
        bias = np.random.randn()
        out_neurons.append(Neuron(weights, bias, activation))
    
    # Randomly disconnect neurons
    for layer in layers:
        for neuron in layer.neurons:
            for other_layer in layers:
                for other_neuron in other_layer.neurons:
                    if np.random.rand() > connectivity:
                        neuron.weights[np.random.randint(len(neuron.weights))] = 0
                        
    nn = NeuralNetwork(layers + [Layer(out_neurons, "Output Layer")])
    
    return nn
