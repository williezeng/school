import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        ok = []
        for Ay in x:
            zz = []
            for A in Ay:
                # H, W
                A = np.pad(A, 0, mode='constant')
                h_new = (A.shape[0] - self.kernel_size) // self.stride + 1
                w_new = (A.shape[1] - self.kernel_size) // self.stride + 1
                output_shape = (h_new, w_new)
                kernel_size = (self.kernel_size, self.kernel_size)
                shape = output_shape + kernel_size
                strides = (A.strides[0] * self.stride, A.strides[1] * self.stride) + A.strides

                as_strided_out = np.lib.stride_tricks.as_strided(
                    A,
                    shape=shape,
                    strides=strides,
                    writeable=False)

                A_w = as_strided_out.reshape(-1, *kernel_size)
                zz.append(A_w.max(axis=(1, 2)).reshape(output_shape))
            ok.append(zz)
        out = np.asarray(ok)
        H_out = out.shape[2]
        W_out = out.shape[3]
            # ###########################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        dx = np.zeros(x.shape)
        m, CHANNELS, HEI, WID = dout.shape
        for i in range(m):
            a_prev = x[i]
            for cc in range(CHANNELS):
                for hh in range(HEI):
                    for ww in range(WID):
                        vert_start = hh * self.stride
                        vert_end = vert_start + self.kernel_size
                        horiz_start = ww * self.stride
                        horiz_end = horiz_start + self.kernel_size
                        a_prev_slice = a_prev[cc, vert_start:vert_end, horiz_start:horiz_end]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dx[i, cc, vert_start:vert_end, horiz_start:horiz_end] += np.multiply(mask, dout[i, cc, hh, ww])
        self.dx = dx
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
