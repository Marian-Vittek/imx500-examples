import torch # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.

## create a "neural network" class by creating a class that inherits from nn.Module.
## In fact, in this case the class is not a neural network at all. We simply return input tensor
## as result

class NeuralNetwork(nn.Module):
    
    def __init__(self): 
        super().__init__() # initialize an instance of the parent class, nn.Model.

    ## forward() is the main function executed inside IMX500. It takes an input image frame (tensor(224,224,3))
    ## and runs it through the "neural network". In this case it does nothing and returns input tensor.
    def forward(self, input): 
        return input

