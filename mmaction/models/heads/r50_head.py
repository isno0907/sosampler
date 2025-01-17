import torch.nn as nn
import torch
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class MaxConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.max(dim=self.dim, keepdim=True)


@HEADS.register_module()
class R50Head(BaseHead):

    def __init__(self,
                 num_classes=200,
                 in_channels=2048,
                 num_neurons=4096,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 consensus=dict(type='MaxConsensus', dim=1),
                 frozen=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.in_channels = in_channels
        self.num_neurons = [num_neurons]

        layers = []
        dim_input = in_channels
        for dim_output in self.num_neurons:
            layers.append(nn.Linear(dim_input, dim_output))
            layers.append(nn.BatchNorm1d(dim_output))
            layers.append(nn.ReLU())
            dim_input = dim_output

        self.layers = nn.Sequential(*layers)
        self.model_output_dim = num_neurons
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_neurons[-1], num_classes)

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        self.consensus_type = consensus_type
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        elif consensus_type == 'MaxConsensus':
            self.consensus = MaxConsensus(**consensus_)

        self.softmax = nn.Softmax(dim=1)
        self.frozen = frozen
    def init_weights(self):
        pass
        
    def forward(self, x, num_segs, probs=None, return_logit=False):
        x = self.avg_pool_2d(x)
        x = x.reshape((-1, num_segs) + x.shape[1:])
        num_batchs = x.shape[0]
        x = x.flatten(start_dim=2)
        x = x.view(-1, self.in_channels)
        cls_score = self.layers(x)
        if probs == None:
            cls_score = cls_score.reshape((num_batchs, num_segs, -1))
            if num_segs != 1:
                cls_score = cls_score.squeeze(dim=1)
            if self.consensus_type == 'MaxConsensus':
                cls_score = self.consensus(cls_score)[0]
            elif self.consensus_type == 'AvgConsensus':
                cls_score = self.consensus(cls_score)
            cls_score = cls_score.squeeze(1)
            cls_score = torch.flatten(cls_score, start_dim=1)
            cls_score = self.fc(cls_score)
            if return_logit:
                return cls_score
            if self.frozen:
                cls_score = self.softmax(cls_score)
            return cls_score
        else:
            cls_score = self.fc(cls_score)
            if self.frozen and not return_logit:
                cls_score = self.softmax(cls_score)
            cls_score = cls_score.reshape((num_batchs, num_segs, -1))
            # sum_tensor = torch.sum(probs, dim=1).unsqueeze(-1)
            cls_score = (cls_score * probs.unsqueeze(-1)).sum(dim=1)

            return cls_score

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        if self.frozen:
            self.layers.eval()
            self.fc.eval()
            for param in self.layers.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = False

@HEADS.register_module()
class R50Head2(BaseHead):
    def __init__(self,
                 num_classes=200,
                 in_channels=2048,
                 num_neurons=4096,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 consensus=dict(type='AvgConsensus', dim=1),
                 frozen=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.in_channels = in_channels
        self.num_neurons = [num_neurons]

        layers = []
        dim_input = in_channels
        for dim_output in self.num_neurons:
            layers.append(nn.Linear(dim_input, dim_output))
            layers.append(nn.BatchNorm1d(dim_output))
            layers.append(nn.ReLU())
            dim_input = dim_output

        self.layers = nn.Sequential(*layers)
        self.model_output_dim = num_neurons
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_neurons[-1], num_classes)

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        self.consensus_type = consensus_type
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        elif consensus_type == 'MaxConsensus':
            self.consensus = MaxConsensus(**consensus_)

        self.softmax = nn.Softmax(dim=1)
        self.frozen = frozen
    def init_weights(self):
        pass
        
    def forward(self, x, num_segs, probs=None, return_logit=False):
        x = self.avg_pool_2d(x)
        x = x.reshape((-1, num_segs) + x.shape[1:])
        num_batchs = x.shape[0]
        x = x.flatten(start_dim=2)
        x = x.view(-1, self.in_channels)
        cls_score = self.layers(x)
        cls_score = self.fc(cls_score)
        cls_score = cls_score.reshape((num_batchs, num_segs, -1))
        
        if num_segs != 1:
            cls_score = cls_score.squeeze(dim=1)
        if self.consensus_type == 'MaxConsensus':
            cls_score = self.consensus(cls_score)[0]
        elif self.consensus_type == 'AvgConsensus':
            cls_score = self.consensus(cls_score)
        cls_score = torch.flatten(cls_score, start_dim=1)
        if return_logit:
            return cls_score
        if self.frozen:
            cls_score = self.softmax(cls_score)
        return cls_score

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        if self.frozen:
            self.layers.eval()
            self.fc.eval()
            for param in self.layers.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = False
