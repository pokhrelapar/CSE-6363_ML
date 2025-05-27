
import numpy as np
from abc import ABC, abstractmethod

#Layer
class Layer(ABC):
    @abstractmethod
    def forward(self, input):
       pass

    @abstractmethod
    def backward(self, input):
       pass


# Linear
class Linear(Layer):

    def __init__(self, input_ftrs, output_ftrs):

        self.input_dim = input_ftrs
        self.output_dim = output_ftrs

        # Xavier init
        limit = np.sqrt(6 / (input_ftrs + output_ftrs))
        self.weights =  np.random.uniform(-limit, limit, (output_ftrs, input_ftrs))
        self.bias = np.zeros((output_ftrs,))

        self.inputs = None
        self.grad_weights = np.zeros_like(self.weights, dtype=np.float64)
        self.grad_bias = np.zeros_like(self.bias, dtype=np.float64)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs,self.weights.T) + self.bias.reshape(1,-1)

    def backward(self, gradient):
        #  gradient w.r.t weights: dL/dW
        self.grad_weights = np.dot(gradient.T, self.inputs)

        #  gradient w.r.t bias: Lf/db
        self.grad_bias = np.sum(gradient, axis=0)

        # gradient w.r.t inputs: dL/dX
        grad_inputs = np.dot(gradient, self.weights)

        return grad_inputs


# Sequential
class Sequential(Layer):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        # Pass input through each layer
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradient):

        # Pass gradient through each layer in reverse order
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def save(self, filename):

        weights ={}

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                weights[f'layer_{i}_weights'] = layer.weights
                weights[f'layer_{i}_biases'] = layer.biases

        np.savez(filename, **weights)

    def load(self, filename):
        params = np.load(filename)

        for i, layer in enumerate(self.layers):
            if hasattr(params, 'weights') and hasattr(params, 'bias'):
                layer.weights = params[f'layer_{i}_weights']
                layer.bias = params[f'layer_{i}_bias']

        print(f'Params loaded from {filename}')

    def __str__(self):
      # Return a string summarizing the layers of the model
      summary = "Sequential Model:\n"
      for i, layer in enumerate(self.layers):
          summary += f"Layer {i + 1}: {layer.__class__.__name__}\n"
          if hasattr(layer, 'weights'):
              summary += f"  Weights: {layer.weights.shape}\n"
          if hasattr(layer, 'bias'):
              summary += f"  Bias: {layer.bias.shape}\n"
      return summary


# Sigmoid
class Sigmoid(Layer):
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # forward pass using sigmoid
        self.output = 1 / (1 + np.exp(-inputs))  # Logistic sigmoid function
        return self.output

    def backward(self, gradient):

        sigmoid_dr = self.output * (1 - self.output)
        return gradient * sigmoid_dr

class Tanh(Layer):
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, gradient):
        tanh_grad = (1 - self.output ** 2)
        return gradient * tanh_grad


# ReLU
class ReLU(Layer):

    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, gradient):
        relu_dr = (self.inputs > 0).astype(float)  # ReLU derivative
        return gradient * relu_dr

# BCE Loss
class BinaryCrossEntropyLoss(Layer):
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

        bce_loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return bce_loss

    def backward(self):
        gradient = (self.predictions - self.targets) / (self.predictions * (1 - self.predictions))
        return gradient
