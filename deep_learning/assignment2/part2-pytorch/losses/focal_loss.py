import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = []
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(cls_num_list)
    per_cls_weights = weights



    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        x,y = input.shape
        labels_one_hot = F.one_hot(target, y).float()
        BCLoss = F.binary_cross_entropy_with_logits(input=input, target=labels_one_hot, reduction="none")
        weights = torch.tensor(self.weight).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, y)

        modulator = torch.exp(-self.gamma * labels_one_hot * input - self.gamma * torch.log(1 + torch.exp(-1.0 * input)))

        loss = modulator * BCLoss
        weighted_loss = weights * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels_one_hot)
        loss = focal_loss


        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
