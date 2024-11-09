import torch # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.

## create a neural network class by creating a class that inherits from nn.Module.
class NeuralNetwork(nn.Module):
    
    def __init__(self): 
        super().__init__() # initialize an instance of the parent class, nn.Model.

    ## forward() is the main function executed inside IMX500. It takes an input image frame (tensor(3,224,224))
    ## and runs it through the "neural network". It returns a tensor which is returned to RPi in metadata.
    def forward(self, input): 
        # It seems that I can not simply return constant "hello world" tensor, I have to 'compute' it from the input
        # otherwise mct issues an error. So:
        
        # reduce the size of input to 1D tensor
        oo = input[:, -1, -1]
        # substract it from itself, hence getting a zero tensor
        oo = torch.sub(oo, oo)
        # prepare another tensor containing the "Hello world" message
        zz = torch.zeros([224])
        text = list("Hello world from IMX500".encode('ascii'))
        zz[0:len(text)] = torch.tensor(text)
        # add the tensor containing "Hello world" message to zeroed input
        oo = torch.add(oo, zz);
        # return the 1D tensor. In general I can return 2D or 3D tensor as well.
        return oo

