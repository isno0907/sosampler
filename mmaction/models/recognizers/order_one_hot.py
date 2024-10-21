import torch
import random
import numpy as np

from ..registry import RECOGNIZERS
from .base import BaseRecognizer
from torch import nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.utils.weight_init import trunc_normal_
import torch.nn.functional as F
from mmcv import ConfigDict
from torch import Tensor

import torchvision.transforms as transforms
from PIL import Image
from einops import rearrange
from itertools import combinations


class PairwiseLoss(nn.Module):
    def __init__(self, total_segment=10, sort='hinge', gamma=0) -> None:
        super(PairwiseLoss, self).__init__()
        self.total_segment = 10
        self.sort = sort
        self.gamma = gamma
        self.topk = None
        self.i_index = []
        self.j_index = []
        for i, j in list(combinations([i for i in range(total_segment)], 2)):
            self.i_index.append(i)
            self.j_index.append(j)
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        delta_input = input[:,self.i_index] - input[:,self.j_index]
        delta_target = target[:, self.i_index] - target[:,self.j_index]
        sign = torch.sign(delta_target)
        delta_target = delta_target * sign
        delta_input = delta_input * sign
        if self.sort == 'hinge':
            result = delta_target * torch.clamp(self.gamma - delta_input, min=0)
            return result.sum(-1).mean()
        elif self.sort == 'exp':
            result = delta_target * torch.exp(self.gamma - delta_input)
            return result.sum(-1).mean()
        elif self.sort == 'log':
            result = delta_target * torch.log(1 + torch.exp(self.gamma - delta_input))
            return result.sum(-1).mean()
        elif self.sort == 'bpr':
            result = -torch.log(torch.sigmoid(delta_input))
            return result.sum(-1).mean()
        else:
            raise("Pairwise Loss: Sort must be in ['hinge', 'exp', 'log', 'bpr] ")

@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        # imgs [b n c t h w]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # imgs [b c t h w]
        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        if self.selected_index:
            imgs = imgs[:, :, :, self.selected_index, ...]

        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            cls_scores = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if hasattr(self, 'neck'):
                    x, _ = self.neck(x)
                cls_score = self.cls_head(x)
                cls_scores.append(cls_score)
                view_ptr += self.max_testing_views
            cls_score = torch.cat(cls_scores)
        else:
            x = self.extract_feat(imgs)
            if hasattr(self, 'neck'):
                x, _ = self.neck(x)
            cls_score = self.cls_head(x)

        cls_score = self.average_clip(cls_score, num_segs)

        return cls_score.cpu().numpy()

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs)

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if hasattr(self, 'neck'):
            x, _ = self.neck(x)

        outs = (self.cls_head(x), )
        return outs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)


@RECOGNIZERS.register_module()
class KDSampler2DRecognizer3D(BaseRecognizer):
    """3D recognizer model framework."""
    def __init__(self,
                 backbone,
                 cls_head,
                 sampler=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_sampler=False,
                 loss='kl',
                 simple=False,
                 embed_dims=768,
                 num_heads=12,
                 num_layers=2,
                 num_segments=10,
                 num_test_segments=1,
                 softmax=False,
                 return_logit=True,
                 temperature=1.0,
                 gamma=0.0,
                 resize_px=None):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        
        if sampler is None:
            self.sampler = None
        self.resize_px=resize_px
        assert loss in ['kl','mse','hinge','exp','log','bpr','bce']
        if loss == 'kl':
            self.loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        elif loss == 'mse':
            self.loss = torch.nn.MSELoss()
        elif loss == 'bce':
            self.loss = torch.nn.BCELoss()
            assert softmax == False
            assert return_logit == False
        else:
            self.loss = PairwiseLoss(total_segment=num_segments, sort=loss, gamma=gamma)
        self.loss_name = loss
        self.num_segments = num_segments
        self.num_test_segments = num_test_segments
        self.softmax = softmax
        self.simple = simple
        self.return_logit = return_logit
        if self.softmax and self.simple:
            raise("softmax and simple cannot be applied simultaneously")
        if self.return_logit and self.simple:
            raise("return_logit and simple cannot be applied simultaneously")
        self.temperature = temperature
        if self.sampler.__class__.__name__ == 'MobileNetV2TSM':
            self.input_dims = 1280
        else:
            self.input_dims = 2048 # 1280 Correction needed
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sample_fc = nn.Linear(self.input_dims, 1)
    
    def sample_forward(self, x_s, num_segs, test_mode=False):
        """Defines getting sample distribution"""
        x_s = torch.flatten(self.avg_pool(x_s), 1) # [N * num_segs, in_channels]
        x_s = x_s.reshape(-1, num_segs, self.input_dims) # [N, num_segs, embed_dims]
        logit = self.sample_fc(x_s).squeeze(-1) # [N, num_segs]
        if not self.return_logit:
            logit = torch.sigmoid(logit)
        if self.softmax:
            logit = F.softmax(logit, -1)
        return logit
    
    def forward_train(self, imgs, labels, **kwargs):
        """
        Defines the computation performed at every call when training.
        If N clip comes as input, then imgs shape is [B, N, C, T, H, W]
        When FormatShape is NCTHW, imgs shape is [B, 1, C, T, H, W]
        """ 
        self.backbone.eval()
        B, N, C, T, H, W = imgs.shape #imgs [B, N, C, T, H, W]
        # imgs = imgs.reshape(B * N * T, C, H, W) #imgs [B * N * T, C, H, W]
        imgs = rearrange(imgs, 'b n c t h w -> (b n) c t h w')
        imgs = imgs.transpose(1, 2).contiguous()
        imgs = imgs.reshape((-1,) + (imgs.shape[-3:]))
        # imgs = rearrange(imgs, 'b t c h w -> (b t) c h w')
        # if self.backbone.__class__.__name__ == 'ResNet3dSlowOnly':
        #     x = False
        x = self.extract_feat(imgs.unsqueeze(2)) # input shape of 3D models must be [b c t h w]. t = 1 with unsqueeze
        if hasattr(self, 'sampler'):
            if self.resize_px is None:
                x_s = self.sampler(imgs).squeeze() # input shape of 2d samplers must be [b*t c h w]
            else:
                x_s = self.sampler(F.interpolate(imgs, size=self.resize_px)).squeeze()
        else:
            x_s = x.clone()
            
        losses = dict()

        if hasattr(self, 'neck'):
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)
        cls_score = self.cls_head(x) # cls_score [B * T, num_classes]
        if not self.return_logit:
            cls_score = cls_score.softmax(-1)
        # cls_score  = rearrange(cls_score, '(b t) c -> b t c', b = B, t = T)
        cls_score = cls_score.reshape(B, T, -1) # [B, T, num_classes]
        gt_labels = labels.squeeze()
        #여기 구현좀 다시 손보자
        # gt_logit = []
        # for i in range(T):
        #     tmp = cls_score[:,i,:].clone() # [B, num_classes]
        #     gt_logit.append(tmp[range(B), gt_labels])
        # gt_logit = torch.stack(gt_logit, -1).to(cls_score.device) # [B, T]
        gt_logit = cls_score[range(B),:,gt_labels] # [B, T]
        if self.softmax:
            gt_logit = F.softmax(gt_logit/self.temperature, -1)
        if self.simple:
            simple_index = gt_logit.topk(3, dim=1)[1]
            batch_index = torch.arange(B).unsqueeze(-1).expand_as(simple_index)
            gt_logit[:] = 0.0
            gt_logit[batch_index, simple_index] = 1.0
            # gt_logit[gt_logit >= 0.5] = 1.0
            # gt_logit[gt_logit < 0.5] = 0.0
        # sampler dist
        logit = self.sample_forward(x_s, T) # [N, num_segs]
        losses[f'{self.loss_name}_loss'] = self.loss(logit, gt_logit)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        # transform = transforms.ToPILImage()
        # if self.selected_index:
        #     imgs = imgs[:, :, :, self.selected_index, ...]
        B, N, C, T, H, W = imgs.shape
        # imgs = imgs.reshape(B * N * T, C, H, W)
        imgs = rearrange(imgs, 'b n c t h w -> (b n t) c h w')
        if self.sampler is not None:
            if self.resize_px is None:
                x_s = self.sampler(imgs).squeeze() # x_s = [B * T, in_channels, h, w]
            else:
                x_s = self.sampler(F.interpolate(imgs, size=self.resize_px)).squeeze() # x_s = [B * N * T, in_channels, h, w]
        else:
            x_s = self.extract_feat(imgs.unsqueeze(2)) # x_s = [B * N * T, in_channels, h, w]
        
        probs = self.sample_forward(x_s, T, test_mode=True) # [N, T]
        sample_index = probs.topk(self.num_test_segments, dim=1)[1]
        sample_index, _ = sample_index.sort(dim=1, descending=False)
        batch_inds = torch.arange(B).unsqueeze(-1).expand_as(sample_index)
        # sample_probs = probs[batch_inds, sample_index]
        # imgs = imgs.reshape(B, T, C, H, W)
        imgs = rearrange(imgs, '(b t) c h w -> b t c h w', b = B, t = T)
        sampled_imgs = imgs[batch_inds, sample_index] # sampled_imgs [B, t, C, H, W], where t is the number of sampled frames
        # sampled_imgs = sampled_imgs.transpose(1,2) # sampled_imgs [B, C, t, H, W]. Timesformer input must be [b c t h w]
        sampled_imgs = rearrange(sampled_imgs, 'b t c h w -> b c t h w')
        # print(f"sampled_imgs = {sampled_imgs}")
        x = self.extract_feat(sampled_imgs)
        cls_score = self.cls_head(x) 

        cls_score = self.average_clip(cls_score, N)

        return cls_score.cpu().numpy()

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs)

    # def forward_dummy(self, imgs):
    #     """Used for computing network FLOPs.

    #     See ``tools/analysis/get_flops.py``.

    #     Args:
    #         imgs (torch.Tensor): Input images.

    #     Returns:
    #         Tensor: Class score.
    #     """
    #     # imgs = imgs.reshape((-1, ) + imgs.shape[2:])
    #     imgs = rearrange(imgs, 'b n c t h w -> (b n t) c h w')
    #     imgs = imgs.unsqueeze(2)
    #     x = self.extract_feat(imgs)

    #     if hasattr(self, 'neck'):
    #         x, _ = self.neck(x)

    #     outs = (self.cls_head(x), )
    #     return outs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)

@RECOGNIZERS.register_module()
class FUllKDSampler2DRecognizer3D(BaseRecognizer):
    """3D recognizer model framework."""
    def __init__(self,
                 backbone,
                 cls_head,
                 sampler=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_sampler=False,
                 loss='kl',
                 simple=False,
                 embed_dims=768,
                 num_heads=12,
                 num_layers=2,
                 num_segments=10,
                 num_classes=200,
                 num_test_segments=1,
                 softmax=False,
                 return_logit=True,
                 temperature=1.0,
                 gamma=0.0,
                 dropout_ratio=0.5,
                 resize_px=None):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        
        if sampler is None:
            self.sampler = None
        self.resize_px=resize_px
        assert loss in ['kl','mse','hinge','exp','log','bpr','bce']
        if loss == 'kl':
            self.loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        elif loss == 'mse':
            self.loss = torch.nn.MSELoss()
        elif loss == 'bce':
            self.loss = torch.nn.BCELoss()
            assert softmax == False
            assert return_logit == False
        else:
            self.loss = PairwiseLoss(total_segment=num_segments, sort=loss, gamma=gamma)
        self.loss_name = loss
        self.num_segments = num_segments
        self.num_test_segments = num_test_segments
        self.softmax = softmax
        self.simple = simple
        self.return_logit = return_logit
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        if self.softmax and self.simple:
            raise("softmax and simple cannot be applied simultaneously")
        if self.return_logit and self.simple:
            raise("return_logit and simple cannot be applied simultaneously")
        self.temperature = temperature
        if self.sampler.__class__.__name__ == 'MobileNetV2TSM':
            self.input_dims = 1280
        else:
            self.input_dims = 2048 # 1280 Correction needed
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.sample_fc = nn.Linear(self.input_dims, num_classes)
    
    def sample_forward(self, x_s, num_segs, test_mode=False):
        """Defines getting sample distribution"""
        x_s = torch.flatten(self.avg_pool(x_s), 1) # [N * num_segs, in_channels]
        if self.dropout is not None:
            x_s = self.dropout(x_s)
        logit = self.sample_fc(x_s) # [N * num_segs, num_classes]
        # x_s = x_s.reshape(-1, num_segs, self.num_classes) # [N, num_segs, num_classes]
        if self.return_logit and not test_mode:
            return logit
        logit = F.softmax(logit, -1)
        return logit
    
    def forward_train(self, imgs, labels, **kwargs):
        """
        Defines the computation performed at every call when training.
        If N clip comes as input, then imgs shape is [B, N, C, T, H, W]
        When FormatShape is NCTHW, imgs shape is [B, 1, C, T, H, W]
        """ 
        self.backbone.eval()
        B, N, C, T, H, W = imgs.shape #imgs [B, N, C, T, H, W]
        # imgs = imgs.reshape(B * N * T, C, H, W) #imgs [B * N * T, C, H, W]
        imgs = rearrange(imgs, 'b n c t h w -> (b n) c t h w')
        imgs = imgs.transpose(1, 2).contiguous()
        imgs = imgs.reshape((-1,) + (imgs.shape[-3:]))
        # imgs = rearrange(imgs, 'b t c h w -> (b t) c h w')
        # if self.backbone.__class__.__name__ == 'ResNet3dSlowOnly':
        #     x = False
        x = self.extract_feat(imgs.unsqueeze(2)) # input shape of 3D models must be [b c t h w]. t = 1 with unsqueeze
        if hasattr(self, 'sampler'):
            if self.resize_px is None:
                x_s = self.sampler(imgs).squeeze() # input shape of 2d samplers must be [b*t c h w]
            else:
                x_s = self.sampler(F.interpolate(imgs, size=self.resize_px)).squeeze()
        else:
            x_s = x.clone()
            
        losses = dict()

        if hasattr(self, 'neck'):
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)
        cls_score = self.cls_head(x) # cls_score [B * T, num_classes]
        if not self.return_logit:
            cls_score = cls_score.softmax(-1)
        # cls_score  = rearrange(cls_score, '(b t) c -> b t c', b = B, t = T)
        # cls_score = cls_score.reshape(B, T, -1) # [B, T, num_classes]
        # gt_labels = labels.squeeze()
        
        # gt_logit = cls_score[range(B),:,gt_labels]
        
        if self.softmax:
            gt_logit = F.softmax(gt_logit/self.temperature, -1)
        
        # sampler dist
        logit = self.sample_forward(x_s, T) # [B * T, num_classes]
        losses[f'{self.loss_name}_loss'] = self.loss(logit, gt_logit)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        # transform = transforms.ToPILImage()
        # if self.selected_index:
        #     imgs = imgs[:, :, :, self.selected_index, ...]
        B, N, C, T, H, W = imgs.shape
        # imgs = imgs.reshape(B * N * T, C, H, W)
        imgs = rearrange(imgs, 'b n c t h w -> (b n t) c h w')
        if self.sampler is not None:
            if self.resize_px is None:
                x_s = self.sampler(imgs).squeeze() # x_s = [B * T, in_channels, h, w]
            else:
                x_s = self.sampler(F.interpolate(imgs, size=self.resize_px)).squeeze() # x_s = [B * N * T, in_channels, h, w]
        else:
            x_s = self.extract_feat(imgs.unsqueeze(2)) # x_s = [B * N * T, in_channels, h, w]
        
        probs = self.sample_forward(x_s, T, test_mode=True) # [B * T, num_classes]
        max_score, max_index = torch.max(probs, -1) # [B * T]
        max_score = max_score.reshape(B, T) # [B, T]
        sample_index = max_score.topk(self.num_test_segments, dim=1)[1]
        sample_index, _ = sample_index.sort(dim=1, descending=False)
        batch_inds = torch.arange(B).unsqueeze(-1).expand_as(sample_index)
        # sample_probs = probs[batch_inds, sample_index]
        # imgs = imgs.reshape(B, T, C, H, W)
        imgs = rearrange(imgs, '(b t) c h w -> b t c h w', b = B, t = T)
        sampled_imgs = imgs[batch_inds, sample_index] # sampled_imgs [B, t, C, H, W], where t is the number of sampled frames
        # sampled_imgs = sampled_imgs.transpose(1,2) # sampled_imgs [B, C, t, H, W]. Timesformer input must be [b c t h w]
        sampled_imgs = rearrange(sampled_imgs, 'b t c h w -> b c t h w')
        # print(f"sampled_imgs = {sampled_imgs}")
        x = self.extract_feat(sampled_imgs)
        cls_score = self.cls_head(x) 

        cls_score = self.average_clip(cls_score, N)

        return cls_score.cpu().numpy()

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs)


    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)
