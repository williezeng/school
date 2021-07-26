import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(2, stride=2)
        # self.flatten = torch.flatten
        self.hidden = torch.nn.Linear(512, 128)
        self.output = nn.Linear(128, 10)
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxp(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
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
