# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, FEEDFORWARD_NETWORK, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import FFN, build_dropout, BaseTransformerLayer
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import digit_version

@TRANSFORMER_LAYER.register_module()
class NormalTransformerLayer(BaseTransformerLayer):
    """Base `TransformerLayer` for vision transformer.
    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                nframes=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.
        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    nframes=nframes,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    nframes=nframes,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@ATTENTION.register_module()
class DividedTemporalAttentionWithNorm(BaseModule):
    """Temporal Attention in Divided Space Time Attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        num_frames (int): Number of frames in the video.
        attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
            0..
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Defaults to 0..
        dropout_layer (dict): The dropout_layer used when adding the shortcut.
            Defaults to `dict(type='DropPath', drop_prob=0.1)`.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
        init_cfg (dict | None): The Config for initialization. Defaults to
            None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

        if digit_version(torch.__version__) < digit_version('1.9.0'):
            kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)

        self.init_weights()

    def init_weights(self):
        constant_init(self.temporal_fc, val=0, bias=0)

    def forward(self, query, key=None, value=None, residual=None, nframes=None, **kwargs):
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        init_cls_token = query[:, 0, :].unsqueeze(1)
        identity = query_t = query[:, 1:, :]

        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        if nframes:
            p, t = pt // nframes, nframes
        else:
            p, t = pt // self.num_frames, self.num_frames

        # res_temporal [batch_size * num_patches, num_frames, embed_dims]
        query_t = self.norm(query_t.reshape(b * p, t, m)).permute(1, 0, 2)
        res_temporal = self.attn(query_t, query_t, query_t)[0].permute(1, 0, 2)
        res_temporal = self.dropout_layer(
            self.proj_drop(res_temporal.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)

        # res_temporal [batch_size, num_patches * num_frames, embed_dims]
        res_temporal = res_temporal.reshape(b, p * t, m)

        # ret_value [batch_size, num_patches * num_frames + 1, embed_dims]
        new_query_t = identity + res_temporal
        new_query = torch.cat((init_cls_token, new_query_t), 1)
        return new_query


@ATTENTION.register_module()
class DividedSpatialAttentionWithNorm(BaseModule):
    """Spatial Attention in Divided Space Time Attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        num_frames (int): Number of frames in the video.
        attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
            0..
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Defaults to 0..
        dropout_layer (dict): The dropout_layer used when adding the shortcut.
            Defaults to `dict(type='DropPath', drop_prob=0.1)`.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
        init_cfg (dict | None): The Config for initialization. Defaults to
            None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        if digit_version(torch.__version__) < digit_version('1.9.0'):
            kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # init DividedSpatialAttentionWithNorm by default
        pass

    def forward(self, query, key=None, value=None, residual=None, nframes=None, **kwargs):
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        identity = query
        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_s = query[:, 1:, :]

        # query_s [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_s.size()
        if nframes:
            p, t = pt // nframes, nframes
        else:
            p, t = pt // self.num_frames, self.num_frames

        # cls_token [batch_size * num_frames, 1, embed_dims]
        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t,
                                                           m).unsqueeze(1)

        # query_s [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = self.norm(query_s).permute(1, 0, 2)
        res_spatial = self.attn(query_s, query_s, query_s)[0].permute(1, 0, 2)
        res_spatial = self.dropout_layer(
            self.proj_drop(res_spatial.contiguous()))

        # cls_token [batch_size, 1, embed_dims]
        cls_token = res_spatial[:, 0, :].reshape(b, t, m)
        cls_token = torch.mean(cls_token, 1, True)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        res_spatial = torch.cat((cls_token, res_spatial), 1)

        new_query = identity + res_spatial
        return new_query


@FEEDFORWARD_NETWORK.register_module()
class FFNWithNorm(FFN):
    """FFN with pre normalization layer.

    FFNWithNorm is implemented to be compatible with `BaseTransformerLayer`
    when using `DividedTemporalAttentionWithNorm` and
    `DividedSpatialAttentionWithNorm`.

    FFNWithNorm has one main difference with FFN:

    - It apply one normalization layer before forwarding the input data to
        feed-forward networks.

    Args:
        embed_dims (int): Dimensions of embedding. Defaults to 256.
        feedforward_channels (int): Hidden dimension of FFNs. Defaults to 1024.
        num_fcs (int, optional): Number of fully-connected layers in FFNs.
            Defaults to 2.
        act_cfg (dict): Config for activate layers.
            Defaults to `dict(type='ReLU')`
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Defaults to 0..
        add_residual (bool, optional): Whether to add the
            residual connection. Defaults to `True`.
        dropout_layer (dict | None): The dropout_layer used when adding the
            shortcut. Defaults to None.
        init_cfg (dict): The Config for initialization. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
    """

    def __init__(self, *args, norm_cfg=dict(type='LN'), **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

    def forward(self, x, residual=None):
        assert residual is None, ('Cannot apply pre-norm with FFNWithNorm')
        return super().forward(self.norm(x), x)
