import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization


    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid= nn.Sigmoid()
        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        #   Initialize the gates in the order above!                                   #
        #   Initialize parameters in the order they appear in the equation!            #                                                              #
        ################################################################################
        
        #i_t: input gate
        self.Wii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        # f_t: the forget gate

        self.Wif = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))
        # g_t: the cell gate
        self.tanh = nn.Tanh()
        self.Wig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))
        
        # o_t: the output gate

        self.Wio = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Who = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        batch_size, time_steps, input_size = x.shape
        h_t, c_t = (
            torch.zeros(batch_size, self.hidden_size).to(x.device),
            torch.zeros(batch_size, self.hidden_size).to(x.device),
        )
        if init_states is not None:
            h_t, c_t = init_states


        for t in range(0, time_steps, 1):
            xt = x[:, t, :]
            i_t = self.sigmoid(xt @ self.Wii + h_t @ self.Whi + self.b_hi)
            f_t = self.sigmoid(xt @ self.Wif + h_t @ self.Whf + self.b_hf)
            g_t = self.tanh(xt @ self.Wig + h_t @ self.Whg + self.b_hg)
            o_t = self.sigmoid(xt @ self.Wio + h_t @ self.Who + self.b_ho)
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * self.tanh(c_t)

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

