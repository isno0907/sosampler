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
        if self.sort == 'ahinge':
            result = torch.clamp(self.gamma + delta_target - delta_input, min=0)
            return result.sum(-1).mean()
        elif self.sort == 'aexp':
            result = torch.exp(self.gamma + delta_target- delta_input)
            return result.sum(-1).mean()
        elif self.sort == 'alog':
            result = torch.log(1 + torch.exp(self.gamma + delta_target - delta_input))
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
class SOSampler2DRecognizer3D(BaseRecognizer):
    """3D recognizer model framework."""
    def __init__(self,
                 backbone,
                 cls_head,
                 sampler=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 label_conf=True,
                 use_sampler=False,
                 loss='kl',
                 ce_loss=False,
                 ft_loss=False,
                 loss_lambda=1.0,
                 simple=False,
                 embed_dims=768,
                 num_heads=12,
                 num_layers=2,
                 num_segments=10,
                 num_test_segments=1,
                 num_classes=200,
                 softmax=False,
                 return_logit=True,
                 temperature=1.0,
                 use_bb_head=False,
                 gamma=0.0,
                 dropout_ratio=0.0,
                 resize_px=None):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        
        if sampler is None:
            self.sampler = None
        self.resize_px=resize_px
        assert loss in ['kl','mse','hinge','exp','log','ahinge','aexp','alog','bpr','bce']
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
        self.ce_loss = ce_loss
        self.ft_loss = ft_loss
        self.loss_lambda = loss_lambda
        self.loss_name = loss
        self.num_segments = num_segments
        self.num_test_segments = num_test_segments
        self.num_classes = num_classes
        self.softmax = softmax
        self.simple = simple
        self.return_logit = return_logit
        self.label_conf = label_conf
        self.use_bb_head = use_bb_head
        if self.softmax and self.simple:
            raise("softmax and simple cannot be applied simultaneously")
        if self.return_logit and self.simple:
            raise("return_logit and simple cannot be applied simultaneously")
        self.temperature = temperature
        if self.sampler.__class__.__name__ in ['MobileNetV2TSM','MobileNetV2','FlexibleMobileNetV2TSM']:
            self.input_dims = 1280
        else:
            self.input_dims = 2048 # 1280 Correction needed
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sample_fc = nn.Linear(self.input_dims, 1)
        if self.ce_loss:
            self.ce = torch.nn.CrossEntropyLoss()
            if self.use_bb_head:
                self.sampler_head = nn.Linear(self.input_dims, self.embed_dims)
            else:
                self.sampler_head = nn.Linear(self.input_dims, self.num_classes)
        if self.ft_loss:
            self.ft = torch.nn.MSELoss()
            self.ft_head = nn.Linear(self.input_dims, self.embed_dims)
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.avg_nframe = []
    
    def sample_forward(self, imgs, num_segs, test_mode=False):
        """Defines getting sample distribution"""
        if self.resize_px is None:
            x_s = self.sampler(imgs, num_segs).squeeze() # input shape of 2d samplers must be [b*t c h w]
        else:
            x_s = self.sampler(F.interpolate(imgs, size=self.resize_px), num_segs).squeeze()
        x_s = torch.flatten(self.avg_pool(x_s), 1) # [N * num_segs, in_channels]
        # x_s = x_s.reshape(-1, num_segs, self.input_dims) # [N, num_segs, embed_dims]
        if self.dropout is not None:
            x_s = self.dropout(x_s)
        logit = self.sample_fc(x_s).squeeze(-1) # [N, num_segs]
        logit = logit.reshape(-1, num_segs)
        if test_mode:
            return x_s, logit
        if not self.return_logit:
            logit = torch.sigmoid(logit)
        if self.softmax:
            logit = F.softmax(logit, -1)
            # logit = F.softmax(logit/self.temperature, -1)
        return x_s, logit
    
    def forward_train(self, imgs, labels, **kwargs):
        """
        Defines the computation performed at every call when training.
        If N clip comes as input, then imgs shape is [B, N, C, T, H, W]
        When FormatShape is NCTHW, imgs shape is [B, 1, C, T, H, W]
        """ 
        
        # if self.backbone.__class__.__name__ == 'ResNet3dSlowOnly':
        #     x = False
        # if hasattr(self, 'sampler'):
        #     if self.resize_px is None:
        #         x_s = self.sampler(imgs).squeeze() # input shape of 2d samplers must be [b*t c h w]
        #     else:
        #         x_s = self.sampler(F.interpolate(imgs, size=self.resize_px)).squeeze()
        # else:
        #     x_s = x.clone()

        self.backbone.eval()
        self.cls_head.eval()
        B, N, C, T, H, W = imgs.shape #imgs [B, N, C, T, H, W]
        imgs = rearrange(imgs, 'b n c t h w -> (b n) c t h w')
        imgs = imgs.transpose(1, 2).contiguous()
        imgs = imgs.reshape((-1,) + (imgs.shape[-3:]))
            
        losses = dict()
        
        x_s, logit = self.sample_forward(imgs, T) # [N, num_segs]
        # x = self.extract_feat(imgs.unsqueeze(2), is_resnet=True) # input shape of 3D models must be [b c t h w]. t = 1 with unsqueeze
        x = self.extract_feat(imgs.unsqueeze(2)) # input shape of 3D models must be [b c t h w]. t = 1 with unsqueeze
        # print(len(x))
        # print(x[-1].shape)
        cls_score = self.cls_head(x) # cls_score [B * T, num_classes]
        # print(cls_score.shape)
        # raise NotImplementedError
        if not self.return_logit:
            cls_score = cls_score.softmax(-1)
        cls_score = cls_score.reshape(B, T, -1) # [B, T, num_classes]
        gt_labels = labels.squeeze()
        if self.label_conf:
            gt_logit = cls_score[range(B),:,gt_labels] # [B, T]
        else:
            gt_logit = cls_score.max(dim=-1)[0]
        if self.softmax:
            gt_logit = F.softmax(gt_logit/self.temperature, -1)
        if self.simple:
            simple_index = gt_logit.topk(3, dim=1)[1]
            batch_index = torch.arange(B).unsqueeze(-1).expand_as(simple_index)
            gt_logit[:] = 0.0
            gt_logit[batch_index, simple_index] = 1.0
            # gt_logit[gt_logit >= 0.3] = 1.0
            # gt_logit[gt_logit < 0.3] = 0.0
        # sampler dist
        if self.ce_loss:
            # x_s = torch.flatten(self.avg_pool(x_s), 1) # [N * num_segs, in_channels]
            # if self.dropout is not None:
            #     x_s = self.dropout(x_s)
            sampler_cls_score = self.sampler_head(x_s) # [N * num_segs, num_classes]
            if self.use_bb_head:
                sampler_cls_score = self.cls_head(sampler_cls_score.reshape(batches, self.embed_dims, num_segs, 1, 1)) # [N, num_segs]
            else:
                sampler_cls_score = sampler_cls_score.reshape(B, T, -1) # [N, num_segs, num_classes]
                sampler_cls_score = sampler_cls_score.mean(1) # [N, num_classes]
            losses['ce_loss'] = self.ce(sampler_cls_score, gt_labels) * (1 - self.loss_lambda)
        if self.ft_loss:
            sampler_ft = self.ft_head(x_s) # [N * num_segs, num_classes]
            gt_ft = x.squeeze()
            # gt_ft = F.relu(x.squeeze()+0.01)-0.01
            losses['ft_loss'] = self.ft(sampler_ft, gt_ft) * (self.loss_lambda / 2)
        losses[f'{self.loss_name}_loss'] = self.loss(logit, gt_logit) * self.loss_lambda
        losses['logit_min'] = logit.min(-1)[0].mean(0)
        losses['logit_max'] = logit.max(-1)[0].mean(0)
        return losses

    def ITS(self, img, logit, T, iter):
        # points = torch.tensor([(2*i+1)/(2*T) for i in range(T)]).to(logit.device)
        for i in range(iter):
            prob = logit.softmax(-1) # N
            cdf = torch.cumsum(prob,-1) # N
            indice = torch.stack([torch.abs(cdf - (2*j+1)/(2*T)).argmin(-1) for j in range(T)])
            indice = torch.unique(indice)
            logit = logit[indice]
            # indice = torch.abs(cdfs - points).argmin(-1).transpose(0,1) # B, T
        
            # indice = torch.stack([tensor.unique(x) for x in indice]).to(logit.device)
            # logit = torch.stack([logit[i, indice[i]] for i in range()])
        self.avg_nframe.append(indice.shape[0])
        # print("avg_nframe")
        sampled_imgs = rearrange(img[indice], 't c h w -> c t h w')
        x = self.extract_feat(sampled_imgs.unsqueeze(0))
        cls_score = self.cls_head(x) 
        
        return cls_score
        
    def Threshold(self, img, logit, thr):
        # prob = logit.softmax(-1)
        indice = torch.where(logit > thr)[0]
        # print(indice)
        if indice.shape[0] == 0:
            indice = logit.topk(1)[1].sort(descending=False)[0]
        elif indice.shape[0] > 7:
            indice = logit.topk(2)[1].sort(descending=False)[0]

        self.avg_nframe.append(indice.shape[0])
        sampled_imgs = rearrange(img[indice], 't c h w -> c t h w')
        x = self.extract_feat(sampled_imgs.unsqueeze(0))
        cls_score = self.cls_head(x) 
        
        return cls_score

            
    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        # transform = transforms.ToPILImage()
        # if self.selected_index:
        #     imgs = imgs[:, :, :, self.selected_index, ...]
        B, N, C, T, H, W = imgs.shape
        # imgs = imgs.reshape(B * N * T, C, H, W)
        # print(f'First imgs shape: {imgs.shape}')
        imgs = rearrange(imgs, 'b n c t h w -> (b n t) c h w') # B 1 
        # if self.sampler is not None:
        #     if self.resize_px is None:
        #         x_s = self.sampler(imgs).squeeze() # x_s = [B * T, in_channels, h, w]
        #     else:
        #         x_s = self.sampler(F.interpolate(imgs, size=self.resize_px)).squeeze() # x_s = [B * N * T, in_channels, h, w]
        # else:
        #     x_s = self.extract_feat(imgs.unsqueeze(2)) # x_s = [B * N * T, in_channels, h, w]
        x_s, logit = self.sample_forward(imgs, T, test_mode=True) # [N, T]
        imgs = rearrange(imgs, '(b n t) c h w -> (b n) t c h w', b = B, t = T)
        if self.num_test_segments != 0:
            # print('self.num_test_segments is not 0') # Yes
            # print(f'Logit shape: {logit.shape}')
            sample_index = logit.topk(self.num_test_segments, dim=1)[1]
            sample_index, _ = sample_index.sort(dim=1, descending=False)

            batch_inds = torch.arange(B).unsqueeze(-1).expand_as(sample_index)
            # sample_probs = probs[batch_inds, sample_index]
            # imgs = imgs.reshape(B, T, C, H, W)
            sampled_imgs = imgs[batch_inds, sample_index] # sampled_imgs [B, t, C, H, W], where t is the number of sampled frames
            # sampled_imgs = sampled_imgs.transpose(1,2) # sampled_imgs [B, C, t, H, W]. Timesformer input must be [b c t h w]
            sampled_imgs = rearrange(sampled_imgs, 'b t c h w -> b c t h w') # 128 3 5 224 224
            # print(f"sampled_imgs = {sampled_imgs}")
            x = self.extract_feat(sampled_imgs) # B 2048 5 7 7
            # print(f'x shape:{x.shape}')
            cls_score = self.cls_head(x) # B classes
            # print(f'x shape:{cls_score.shape}')

            cls_score = self.average_clip(cls_score, N) # B classes
            # print(f'x shape:{cls_score.shape}')
            # raise NotImplementedError
        else:
            cls_score = []
            for i in range(B):
                # cls_score.append(self.ITS(imgs[i],logit[i], 10, 1))
                cls_score.append(self.Threshold(imgs[i],logit[i], .5))
            cls_score = torch.stack(cls_score)
            cls_score = self.average_clip(cls_score, N)
        return cls_score.cpu().detach().numpy()

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

        
