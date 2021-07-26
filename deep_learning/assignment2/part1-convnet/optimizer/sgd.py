from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                prev_v = self.grad_tracker[idx]["dw"]
                velocity = (prev_v*self.momentum) - (self.learning_rate*m.dw)
                m.weight = m.weight + velocity
                self.grad_tracker[idx]["dw"] = velocity
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                pass
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                prev_b = self.grad_tracker[idx]["db"]

                bias = (prev_b*self.momentum) - (self.learning_rate*m.db)
                m.bias = m.bias + bias
                self.grad_tracker[idx]["db"] = bias
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                pass
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
