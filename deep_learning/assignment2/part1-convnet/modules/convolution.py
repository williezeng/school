import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        N, C, H, W = x.shape
        F, channel_holder, HH, WW = self.weight.shape
        NEW_H = H + (2 * self.padding)
        NEW_W = W + (2 * self.padding)
        numh = NEW_H - HH
        numw = NEW_W - WW

        HHH = numh // self.stride + 1
        WWW = numw // self.stride + 1
        strides = (NEW_H * NEW_W, NEW_W, 1, C * NEW_H * NEW_W, self.stride * NEW_W, self.stride)
        shape = (C, HH, WW, N, HHH, WWW)
        strides = x.itemsize * np.array(strides)


        x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols.shape = (C * HH * WW, N * HHH * WWW)

        results = self.weight.reshape(F, -1).dot(x_cols) + self.bias.reshape(-1, 1)
        results.shape = (F, N, HHH, WWW)
        out = results.transpose(1, 0, 2, 3)


        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        N, C, H, W = x.shape
        F, C, HH, WW = self.weight.shape

        try:
            self.db = np.sum(dout, axis=(0,2,3))
        except:
            import pdb
            pdb.set_trace()
        self.dw = np.zeros((F, C, HH, WW))

        padded_x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        padded_dx = np.zeros(padded_x.shape)
        max_Hrange = H - HH + 2 * self.padding + 1
        max_Wrange = H - WW + 2 * self.padding + 1
        for i in range(N):  # images
            for j in range(F):
                for hh in range(0, max_Hrange, self.stride):     # go through h,w
                    for ww_iterator in range(0, max_Wrange, self.stride):
                        new_H = int(hh / self.stride)
                        new_W = int(ww_iterator / self.stride)
                        filter = self.weight[j, :, :, :]
                        padded_dx[i, :, hh:(hh + HH), ww_iterator:(ww_iterator + WW)] += filter * dout[i, j, new_H, new_W]
                        # padded_dx[i, :, hh:(hh + HH), ww_iterator:(ww_iterator + WW)] += filter * dout[new_H, new_W]

                        patch = padded_x[i, :, hh:(hh + HH), ww_iterator:(ww_iterator + WW)]
                        self.dw[j, :, :, :] += patch * dout[i, j, int(hh / self.stride), int(ww_iterator / self.stride)]
                        # self.dw[j, :, :, :] += patch * dout[int(hh / self.stride), int(ww_iterator / self.stride)]

        self.dx = padded_dx[:, :, self.padding:(self.padding + H), self.padding:(self.padding + W)]


        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
