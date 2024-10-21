from ..registry import BACKBONES, SAMPLER
from .mobilenet_v2 import InvertedResidual, MobileNetV2
from .resnet_tsm import TemporalShift
import torch.nn.functional as F
from ...utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.utils import print_log
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv import ConfigDict
import torch
import torch.nn as nn

@BACKBONES.register_module()
@SAMPLER.register_module()
class MobileNetV2TSM(MobileNetV2):
    """MobileNetV2 backbone for TSM.

    Args:
        num_segments (int): Number of frame segments. Default: 8.
        is_shift (bool): Whether to make temporal shift in reset layers.
            Default: True.
        shift_div (int): Number of div for shift. Default: 8.
        **kwargs (keyword arguments, optional): Arguments for MobilNetV2.
    """

    def __init__(self, num_segments=8, is_shift=True, shift_div=8, **kwargs):
        super().__init__(**kwargs)
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div

    def make_temporal_shift(self):
        """Make temporal shift for some layers."""
        for m in self.modules():
            if isinstance(m, InvertedResidual) and \
                    len(m.conv) == 3 and m.use_res_connect:
                m.conv[0] = TemporalShift(
                    m.conv[0],
                    num_segments=self.num_segments,
                    shift_div=self.shift_div)
    
    def _inner_forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def forward(self, x):

        ret = self._inner_forward(x)

        if self.is_sampler:
            ret = self.avg_pool(ret)
            ret = ret.squeeze()
            ret = ret.reshape((-1, self.total_segments, self.out_channel)) # here!!
            logit = self.logit(ret).squeeze(-1)
            if self.gumbel:                
                if self.training:
                    probs = F.gumbel_softmax(logit, tau=self.gumbel_tau, hard=self.gumbel_hard, dim=1)
                else:
                    probs = F.softmax(logit, dim=1)
            else:
                probs = F.softmax(logit, dim=1)
            
            return probs

        return ret

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if self.pretrained is not None and 'tsm' not in self.pretrained:
            super().init_weights()

        if self.is_shift:
            self.make_temporal_shift()

        if self.pretrained is not None and 'tsm' in self.pretrained:
            if isinstance(self.pretrained, str):
                logger = get_root_logger()
                print_log(f'log from {self.pretrained}', logger=logger)
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)

@BACKBONES.register_module()
@SAMPLER.register_module()
class FlexibleMobileNetV2TSM(MobileNetV2):
    """MobileNetV2 backbone for TSM.

    Args:
        num_segments (int): Number of frame segments. Default: 8.
        is_shift (bool): Whether to make temporal shift in reset layers.
            Default: True.
        shift_div (int): Number of div for shift. Default: 8.
        **kwargs (keyword arguments, optional): Arguments for MobilNetV2.
    """

    def __init__(self, num_segments=8, is_shift=True, shift_div=8, embed_dims=1280, num_heads=20, **kwargs):
        super().__init__(**kwargs)
        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.embed_dims = embed_dims

        if self.is_sampler and not self.is_shift:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_segments, self.embed_dims))
            _transformerlayers_cfg = [
                    dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                batch_first=True,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=0.1))
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=embed_dims,
                            feedforward_channels=embed_dims * 4,
                            num_fcs=2,
                            act_cfg=dict(type='GELU'),
                            dropout_layer=dict(
                                type='DropPath', drop_prob=0.1)),
                        operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                        norm_cfg=dict(type='LN', eps=1e-6),
                        batch_first=True)
                ]
            transformer_layers = ConfigDict(
                dict(
                    type='TransformerLayerSequence',
                    transformerlayers=_transformerlayers_cfg,
                    num_layers=1))
            self.transformer_layers = build_transformer_layer_sequence(transformer_layers)


    def make_temporal_shift(self):
        """Make temporal shift for some layers."""
        for m in self.modules():
            if isinstance(m, InvertedResidual) and \
                    len(m.conv) == 3 and m.use_res_connect:
                m.conv[0] = TemporalShift(
                    m.conv[0],
                    num_segments=self.num_segments,
                    shift_div=self.shift_div)
    
    def _inner_forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def forward(self, x, num_seg):
        if self.is_shift:
            for m in self.modules():
                if isinstance(m, InvertedResidual) and len(m.conv) == 3 and m.use_res_connect:
                    m.conv[0].num_segments=num_seg

        ret = self._inner_forward(x)

        if self.is_sampler:
            ret = self._sampling_forward(ret, num_seg)

        return ret
    
    def _sampling_forward(self, ret, num_seg):
        if self.is_shift:
            ret = self.avg_pool(ret)
            ret = ret.squeeze()
            ret = ret.reshape((-1, num_seg, self.out_channel)) # here!!
            logit = self.logit(ret).squeeze(-1)
        else:
            ret = self.avg_pool(ret)
            ret = ret.squeeze()
            ret = ret.reshape((-1, num_seg, self.out_channel))
            ret = ret + self.pos_embed[:,:num_seg]
            ret = self.transformer_layers(ret,None,None)
            logit = self.logit(ret).squeeze(-1)
        if self.gumbel:                
            if self.training:
                probs = F.gumbel_softmax(logit, tau=self.gumbel_tau, hard=self.gumbel_hard, dim=1)
            else:
                probs = F.softmax(logit, dim=1)
        else:
            probs = F.softmax(logit, dim=1)
        
        return probs

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if not self.is_shift:
            trunc_normal_(self.pos_embed, std=.02)
        if self.pretrained is not None and 'tsm' not in self.pretrained:
            super().init_weights()

        if self.is_shift:
            self.make_temporal_shift()

        if self.pretrained is not None and 'tsm' in self.pretrained:
            if isinstance(self.pretrained, str):
                logger = get_root_logger()
                print_log(f'log from {self.pretrained}', logger=logger)
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
