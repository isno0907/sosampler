import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import AvgConsensus, BaseHead

# class WeightedConsensus(nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, x, w):
#         """Defines the computation performed at every call."""
        
#         return (x@w)/w.sum()

@HEADS.register_module()
class TSMHead(BaseHead):
    """Class head for TSM.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        num_segments (int): Number of frame segments. Default: 8.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.80,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 frozen=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool
        self.frozen = frozen

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None
        if self.frozen:
            self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, num_segs=1, return_logit=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Useless in TSMHead. By default, `num_segs`
                is equal to `clip_len * num_clips * num_crops`, which is
                automatically generated in Recognizer forward phase and
                useless in TSM models. The `self.num_segments` we need is a
                hyper parameter to build TSM models.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]
        x = torch.flatten(x, 1)
        # [N * num_segs, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc_cls(x) # [N * num_segs, num_classes]
        # if return_logit:
        #     return cls_score
        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            # cls_score = cls_score.view((-1, self.num_segments // 2) +
            #                            cls_score.size()[1:])
            cls_score = cls_score.view((-1, num_segs // 2) +
                                       cls_score.size()[1:])                           
        else:
            # [N, num_segs, num_classes]
            # cls_score = cls_score.view((-1, self.num_segments) +
            #                            cls_score.size()[1:])
            cls_score = cls_score.view((-1, num_segs) +
                                       cls_score.size()[1:])
        # [N, 1, num_classes]
        cls_score = self.consensus(cls_score)
        if self.frozen and not return_logit:
            cls_score = self.softmax(cls_score)
        # [N, num_classes]
        return cls_score.squeeze(1)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        if self.frozen:
            self.fc_cls.eval()
            for param in self.fc_cls.parameters():
                param.requires_grad = False
