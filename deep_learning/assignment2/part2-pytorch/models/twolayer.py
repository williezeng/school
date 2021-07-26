import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, num_classes)



        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()

        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):

        flattened_x = torch.flatten(x, start_dim=1)
        yy = self.hidden(flattened_x)

        x = self.sigmoid(yy)

        x = self.linear(x)
        out = x
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
