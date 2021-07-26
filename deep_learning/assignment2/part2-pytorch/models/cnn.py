import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.conv = nn.Conv2d(3, 32, 7, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(2, stride=2)
        # self.flatten = torch.flatten
        self.hidden = torch.nn.Linear(5408, 128)
        self.output = nn.Linear(128, 10)

        # # Output layer, 10 units - one for each digit
        # self.output = nn.Linear(256, 10)
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = []
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxp(x)

        flattened_x = torch.flatten(x,1)
        x = self.hidden(flattened_x)
        x = self.output(x)
        outs = x
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
