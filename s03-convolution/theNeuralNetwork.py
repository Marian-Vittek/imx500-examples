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
        # called 'mini-batch' or such. So take it into account.
        bw = input[:,0] * 0.2989 + input[:,1] * 0.5870 + input[:,2] * 0.1140
        # get fixed convolution matrix (or kernel). This one shall detect horizontal edges
        # in the image
        weights = torch.tensor([[[
            [2.,2.,2.,2.,2.],
            [1.,1.,1.,1.,1.],
            [0.,0.,0.,0.,0.],
            [-1.,-1.,-1.,-1.,-1.],
            [-2.,-2.,-2.,-2.,-2.],
        ]]])
        # translate grayscale tensor to the format expected by conv2d
        bw = bw[:, None, :, :]
        # apply convolution
        convoluted = F.conv2d(bw, weights, stride=1, padding=2)
        # both maximal positive and negative values detect an edge, so get absolute value of convoluted tensor
        convoluted_abs = torch.abs(convoluted)
        # filter out values which are too small. Unfortunately, for some reason, I can not get the maximum
        # value of a tensor, torch.max(output) produces an error, so guess it as a constant
        maxval = 20
        # substract the filter value
        pre_filtered = torch.abs(convoluted_abs) - (2*maxval/3)
        # get only positive values
        filtered = F.relu(pre_filtered)
        # negate the output, so that background is white and edges are black in the visualization
        output = - filtered
        # return tensor containing detected edges
        return output

