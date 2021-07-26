from models.naive.LSTM import LSTM

# Just run this block. Please do not modify the following code.
import math
import time
import numpy as np
import csv
import torch


# Pytorch package
import torch
import torch.nn as nn
import torch.optim as optim

# Torchtest package
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# Tqdm progress bar
from tqdm import tqdm_notebook, tqdm

# Code provide to you for training and evaluation
from models.naive.RNN import VanillaRNN

from utils import train, evaluate, set_seed_nb, unit_test_values
from models.seq2seq.Seq2Seq import Seq2Seq
from models.seq2seq.Encoder import Encoder
from models.seq2seq.Decoder import Decoder

set_seed_nb()
embedding_size = 32
hidden_size = 32
input_size = 8
output_size = 8
batch, seq = 1, 2

encoder = Encoder(input_size, embedding_size, hidden_size, hidden_size)
decoder = Decoder(embedding_size, hidden_size, hidden_size, output_size)

seq2seq = Seq2Seq(encoder, decoder, 'cpu')
x_array = np.random.rand(batch, seq) * 10
x = torch.LongTensor(x_array)
expected_out = unit_test_values('seq2seq')

out = seq2seq.forward(x)

print('Close to out: ', expected_out.allclose(out, atol=1e-4))

