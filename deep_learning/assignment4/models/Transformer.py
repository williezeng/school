# Code by Sarah Wiegreffe (saw@gatech.edu)
# Fall 2019

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        '''
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        '''        
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # This should take 1-2 lines.                                                #
        # Initialize the word embeddings before the positional encodings.            #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.word = nn.Embedding(self.input_size, self.hidden_dim)
        self.positional = nn.Embedding(self.max_length, self.hidden_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        # fully connect 1
        self.fc1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        # Relu btwn the twoo
        self.relu = nn.ReLU()
        # fully connect 2
        self.fc2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.final_linear = nn.Linear(self.hidden_dim, self.output_size)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        '''
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups. 

        :returns: the model outputs. Should be normalized scores of shape (N,1).
        '''

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling ClassificationTransformer class methods here.  #
        #############################################################################
        emb = self.embed(inputs)
        outputs = self.final_layer(self.feedforward_layer(self.multi_head_attention(emb)))
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
        N,T = inputs.shape
        token = self.word(inputs)
        vocab_size_tensors = torch.arange(0, T, out=torch.LongTensor()).repeat(N,1)
        positional = self.positional(vocab_size_tensors)
        embeddings = token + positional
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        k1 = self.k1(inputs)
        v1 = self.v1(inputs)
        q1 = self.q1(inputs)
        k2 = self.k2(inputs)
        q2 = self.q2(inputs)
        v2 = self.v2(inputs)
        # attention 1
        sqr_root = np.sqrt(self.dim_k)
        Qx = torch.bmm(q1, k1.permute(0, 2, 1))
        softmax = self.softmax(Qx/sqr_root)
        A1 = torch.bmm(softmax,v1)

        # attention 2
        Qx = torch.bmm(q2, k2.permute(0, 2, 1))
        softmax = self.softmax(Qx/sqr_root)
        A2 = torch.bmm(softmax,v2)
        multi_head_attention = torch.cat((A1,A2), dim=2)
        outputs = self.attention_head_projection(multi_head_attention)
        outputs = self.norm_mh(outputs + inputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        fc = self.fc1(inputs)
        relu = self.relu(fc)
        outputs = self.fc2(relu)
        outputs = self.norm_mh(outputs + inputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        #############################################################################
        outputs = self.final_linear(inputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
