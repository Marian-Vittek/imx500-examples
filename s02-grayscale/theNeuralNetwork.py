import torch # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
import torchvision as tv

## create a "neural network" class by creating a class that inherits from nn.Module.
class NeuralNetwork(nn.Module):
    
    def __init__(self): 
        super().__init__() # initialize an instance of the parent class, nn.Model.

    def forward(self, input):
        # translate to grayscale using the formula R*0.2989 + G*0.5870 + B*0.1140
        # actually, it seems that the input has 4 dimensions, the first one is something
        # called 'mini-batch' or such. Then follows rgb image
        output = input[:,0] * 0.2989 + input[:,1] * 0.5870 + input[:,2] * 0.1140
        return output

