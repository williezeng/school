import random

import torch
import torch.nn as nn
import torch.optim as optim

# import custom models



class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, source, out_seq_len = None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """
        wtf = source
        batch_size = source.shape[0]
        seq_len = source.shape[1]
        if out_seq_len is None:
            out_seq_len = seq_len
        outputs = torch.zeros(batch_size, out_seq_len, self.decoder.output_size).to(self.device)
        output, hidden = self.encoder(source)
        trg = source[:,:1]
        # Get the last hidden representation from the encoder
        if type(hidden) is not tuple:
            holder = hidden[:,-1,:]
            real_hidden = holder.unsqueeze(0).repeat(hidden.shape[0], hidden.shape[1], 1)

        else:
            hidden, cell = hidden
            holder = hidden[:, -1, :]
            real_hiddenh = holder.unsqueeze(0).repeat(hidden.shape[0], hidden.shape[1], 1)
            cholder = cell[:, -1, :]
            real_hiddenc = cholder.unsqueeze(0).repeat(cell.shape[0], cell.shape[1], 1)
            real_hidden = (real_hiddenh, real_hiddenc)

        for i in range(0, seq_len):
            output, replace_hidden = self.decoder(trg, real_hidden)
            # real_hidden = replace_hidden
            outputs[:, i] = output
            trg = output.argmax(axis=1).unsqueeze(0)
            if type(replace_hidden) is not tuple:
                real_hidden = replace_hidden[0][0].unsqueeze(0).unsqueeze(0)
            else:
                real_hidden = (replace_hidden[0][0][0].unsqueeze(0).unsqueeze(0), replace_hidden[1][0][0].unsqueeze(0).unsqueeze(0))


        # eq2seq output shape was [2, 8], and the test passes even though it should really be [1, 2, 8].
        # You have to get to [1, 2, 8] by factoring in the batch_size
         #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden weights being fed into the decoder   #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs



        

