from ..registry import RECOGNIZERS
from .base import BaseRecognizer
from torch import nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.utils.weight_init import trunc_normal_
import torch.nn.functional as F
from mmcv import ConfigDict
import torch

from torch import Tensor
import torchvision.transforms as transforms
from PIL import Image
from einops import rearrange
from itertools import combinations

class PairwiseLoss(nn.Module):
    def __init__(self, total_segment=10, sort='hinge', gamma=0) -> None:
        super(PairwiseLoss, self).__init__()
        self.total_segment = total_segment
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
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, labels.squeeze())
            x = x.squeeze(2)
            num_segs = 1
            losses.update(loss_aux)

        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, num_segs)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)

        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        # outs = (self.cls_head(x, num_segs), )
        # return outs
        return x

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)

@RECOGNIZERS.register_module()
class SOSampler2DRecognizer2D(BaseRecognizer):
    """2D recognizer model framework."""
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
                 num_segments=10,
                 num_classes=200,
                 num_test_segments=6,
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
        self.label_conf = label_conf
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
        self.use_bb_head = use_bb_head
        self.embed_dims = 2048
        if self.softmax and self.simple:
            raise("softmax and simple cannot be applied simultaneously")
        if self.return_logit and self.simple:
            raise("return_logit and simple cannot be applied simultaneously")
        self.temperature = temperature
        if self.sampler.__class__.__name__ in ['MobileNetV2', 'MobileNetV2TSM', 'FlexibleMobileNetV2TSM']:
            self.input_dims = 1280
        else:
            self.input_dims = self.embed_dims # 1280 Correction needed
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
    
    def sample_forward(self, x_s, num_segs, test_mode=False):
        """Defines getting sample distribution"""
        x_s = torch.flatten(self.avg_pool(x_s), 1) # [N * num_segs, in_channels]
        
        ################### mean minus #########################
        
        # x_s = x_s.reshape(-1, num_segs, self.input_dims) # [N, num_segs, embed_dims]
        # x_s = abs(x_s - x_s.mean(1,keepdim=True))
        # x_s = x_s.reshape(-1, self.input_dims) # [N * num_segs, embed_dims]
        
        ########################################################
        
        if self.dropout is not None:
            x_s = self.dropout(x_s)
        logit = self.sample_fc(x_s).squeeze(-1) # [N * num_segs]
        logit = logit.reshape(-1, num_segs) # [N, num_segs]
        if not self.return_logit:
            logit = torch.sigmoid(logit)
        if self.softmax:
            logit = F.softmax(logit, -1)
        return logit

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        self.backbone.eval()
        self.cls_head.eval()
        batches = imgs.shape[0] #imgs [N, num_segs, 3, H, W]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) #imgs [N * num_segs, 3, H, W]
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs) # [N * num_segs, in_channels, h, w]
        if self.sampler is not None:
            if self.resize_px is None:
                x_s = self.sampler(imgs, num_segs).squeeze()
            else:
                x_s = self.sampler(F.interpolate(imgs, size=self.resize_px), num_segs).squeeze()
        else:
            x_s = x.clone()

        # making gt label
        # x = torch.flatten(self.cls_head.avg_pool(x), 1) # [N * num_segs, in_channels]        
        # cls_score = self.cls_head.fc_cls(x) # [N * num_segs, num_classes]
        cls_score = self.cls_head(x, 1, return_logit=True)
        if not self.return_logit:
            cls_score = cls_score.softmax(-1)
        cls_score = cls_score.reshape(batches, num_segs, -1) # [N, num_segs, num_classes]
        gt_labels = labels.squeeze()
        
        if self.label_conf:
            gt_logit = cls_score[range(batches),:,gt_labels] # [B, T]
        else:
            gt_logit = cls_score.max(dim=-1)[0]
        
        # cls_score = cls_score.reshape(-1, self.num_classes)
        # gt_logit = cls_score[range(batches * num_segs), cls_score.argmax(-1)].reshape(batches, num_segs) # [B, T]
        # cls_score = cls_score.reshape(batches, num_segs, self.num_classes)
        if self.softmax:
            gt_logit = F.softmax(gt_logit/self.temperature, -1)
        if self.simple:
            simple_index = gt_logit.topk(3, dim=1)[1]
            batch_index = torch.arange(batches).unsqueeze(-1).expand_as(simple_index)
            gt_logit[:] = 0.0
            gt_logit[batch_index, simple_index] = 1.0
            # gt_logit[gt_logit >= 0.3] = 1.0
            # gt_logit[gt_logit < 0.3] = 0.0
        # sampler dist
        gt_logit = gt_logit.detach()
        logit = self.sample_forward(x_s, num_segs) # [N, num_segs]
        if self.ce_loss:
            flt_x_s = torch.flatten(self.avg_pool(x_s), 1) # [N * num_segs, in_channels]
            if self.dropout is not None:
                flt_x_s = self.dropout(flt_x_s)
            sampler_cls_score = self.sampler_head(flt_x_s) # [N * num_segs, num_classes]
            # print("sampler input = ", sampler_cls_score.shape)
            # print("backbone input = ", x.shape)
            if self.use_bb_head:
                sampler_cls_score = self.cls_head(sampler_cls_score.reshape(batches*num_segs, self.embed_dims, 1, 1), num_segs) # [N, num_segs]
            else:
                sampler_cls_score = sampler_cls_score.reshape(batches, num_segs, -1) # [N, num_segs, num_classes]
                sampler_cls_score = sampler_cls_score.mean(1) # [N, num_classes]
            losses['ce_loss'] = self.ce(sampler_cls_score, gt_labels) * (1 - self.loss_lambda)
        ###
        if self.ft_loss:
            flt_x_s = torch.flatten(self.avg_pool(x_s), 1) # [N * num_segs, in_channels]
            if self.dropout is not None:
                flt_x_s = self.dropout(flt_x_s)
            sampler_ft = self.ft_head(flt_x_s) # [N * num_segs, num_classes]
            gt_ft = x.squeeze()
            # gt_ft = F.relu(x.squeeze()+0.01)-0.01
            losses['ft_loss'] = self.ft(sampler_ft, gt_ft) * (self.loss_lambda / 2)
        losses[f'{self.loss_name}_loss'] = self.loss(logit, gt_logit) * self.loss_lambda
        ###
        losses['logit_min'] = logit.min(-1)[0].mean(0)
        losses['logit_max'] = logit.max(-1)[0].mean(0)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        # x = self.extract_feat(imgs) # [N * num_segs, in_channels, h, w]
        if self.sampler is not None:
            if self.resize_px is None:
                x_s = self.sampler(imgs, num_segs).squeeze()
            else:
                x_s = self.sampler(F.interpolate(imgs, size=self.resize_px), num_segs).squeeze()
        else:
            x_s = self.extract_feat(imgs) # [N * num_segs, in_channels, h, w]
        
        
        probs = self.sample_forward(x_s, num_segs, test_mode=True) # [N, num_segs]
        sample_index = probs.topk(self.num_test_segments, dim=1)[1]
        sample_index, _ = sample_index.sort(dim=1, descending=False)
        batch_inds = torch.arange(batches).unsqueeze(-1).expand_as(sample_index)
        # sample_probs = probs[batch_inds, sample_index]
        imgs = imgs.reshape((batches, -1) + (imgs.shape[-3:]))
        sampled_imgs = imgs[batch_inds, sample_index]
        sampled_imgs = sampled_imgs.reshape((-1, ) + sampled_imgs.shape[2:])

        x = self.extract_feat(sampled_imgs)

        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, self.num_test_segments)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)

        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    # def forward_dummy(self, imgs):
    #     """Used for computing network FLOPs.

    #     See ``tools/analysis/get_flops.py``.

    #     Args:
    #         imgs (torch.Tensor): Input images.

    #     Returns:
    #         Tensor: Class score.
    #     """
    #     batches = imgs.shape[0]
    #     imgs = imgs.reshape((-1, ) + imgs.shape[2:])
    #     num_segs = imgs.shape[0] // batches

    #     x = self.extract_feat(imgs)
    #     if hasattr(self, 'neck'):
    #         x = [
    #             each.reshape((-1, num_segs) +
    #                          each.shape[1:]).transpose(1, 2).contiguous()
    #             for each in x
    #         ]
    #         x, _ = self.neck(x)
    #         x = x.squeeze(2)
    #         num_segs = 1

    #     # outs = (self.cls_head(x, num_segs), )
    #     # return outs
    #     return x

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)
