import torch
import torch.nn.functional as F

import numpy as np

from ..registry import RECOGNIZERS
from .base import BaseRecognizer

from itertools import permutations
import math
from ...core import top_k_accuracy, top_k_recall

def delete(arr: torch.Tensor, ind: list, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if i not in ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)

def SmoothedL1HingeLoss(x, gamma=1.0):
    batch_size = x.shape[0]
    # output = torch.zeros_like(x)
    # for i in range(batch_size):
    # if x >= 1-gamma:
    positive = 1 / (2*gamma) * torch.maximum(torch.zeros_like(x),1-x)**2
    negative = 1 - gamma/2 - x
    p_index = x >= 1-gamma
    n_index = x < 1-gamma
    x[p_index] = positive[p_index]
    x[n_index] = negative[n_index]
    return x

Tensor = torch.Tensor

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, topk: int = 1, eps: float = 1e-10, dim: int = -1) -> Tensor:

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        # index = y_soft.max(dim, keepdim=True)[1]
        index = torch.topk(y_soft, topk, dim)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

@RECOGNIZERS.register_module()
class Sampler2DRecognizer3D(BaseRecognizer):

    def __init__(self,
                 sampler,
                 backbone,
                 cls_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 bp_mode='gradient_policy',
                 calc_mode='all',
                 num_segments=4,
                 num_test_segments=None,
                 use_sampler=False,
                 resize_px=None,
                 explore_rate=1.,
                 num_clips=1,
                 shuffle=False):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        self.resize_px = resize_px
        self.shuffle = shuffle
        self.num_segments = num_segments
        self.bp_mode = bp_mode
        self.num_clips = num_clips
        self.calc_mode = calc_mode
        assert bp_mode in ['gradient_policy', 'tsn', 'random', 'max']
        assert calc_mode in ['approximate', 'all']
        if self.num_segments <= 8:
            self.permute_index = list(permutations(list(range(self.num_segments)), self.num_segments))
            self.index_length = len(self.permute_index)
        if num_test_segments is None:
            self.num_test_segments = num_segments
        else:
            self.num_test_segments = num_test_segments
        self.explore_rate = explore_rate

    def sample(self, imgs, probs, test_mode=False, bp_mode='gradient_policy'):

        if test_mode:
            num_batches, original_segments = probs.shape
            sample_index = probs.topk(self.num_test_segments, dim=1)[1]
            if not self.shuffle:
                sample_index, _ = sample_index.sort(dim=1, descending=False)
            batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
            sample_probs = probs[batch_inds, sample_index]
            distribution = probs
            policy = None
        else:
            if bp_mode == 'gradient_policy':
                num_batches, original_segments = probs.shape
                sample_index = torch.multinomial(probs, self.num_segments, replacement=False)
                batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
                sample_probs = probs[batch_inds, sample_index]
                distribution = probs
                policy = None
                if not self.shuffle:
                    sample_index = sample_index.sort(dim=1)[0]
            elif bp_mode == 'max':
                sample_probs, sample_index = probs.topk(self.num_segments, dim=1)
                sample_index, _ = sample_index.sort(dim=1, descending=False)
                distribution = probs
                policy = None
            elif bp_mode == 'tsn':
                num_batches, original_segments = probs.shape
                num_len = original_segments // self.num_segments
                base_offset = torch.linspace(0, original_segments - num_len,
                                             steps=self.num_segments, dtype=int).repeat(num_batches, 1)
                rand_shift = np.random.randint(num_len, size=(num_batches, self.num_segments))
                sample_index = base_offset + rand_shift
                distribution = probs
                policy = None
                batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
                sample_probs = probs[batch_inds, sample_index]
            elif bp_mode == 'random':
                num_batches, original_segments = probs.shape
                sample_index = []
                for _ in range(num_batches):
                    sample_index.append(np.random.choice(original_segments, self.num_segments, replace=False))
                sample_index = torch.tensor(np.stack(sample_index))
                sample_index = sample_index.sort(dim=1)[0]
                distribution = probs
                policy = None
                batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
                sample_probs = probs[batch_inds, sample_index]

        # num_batches, num_segments
        num_batches = sample_index.shape[0]
        batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)
        selected_imgs = imgs[batch_inds, sample_index]
        return selected_imgs, distribution, policy, sample_index, sample_probs

    def forward_sampler(self, imgs, num_batches, test_mode=False, **kwargs):
        self.backbone.eval()
        self.cls_head.eval()
        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            view_ptr = 0
            probs = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                prob = self.sampler(batch_imgs)
                probs.append(prob)
                view_ptr += self.max_testing_views
            probs = torch.cat(probs)
        else:
            if self.resize_px is None:
                probs = self.sampler(imgs)
            else:
                probs = self.sampler(F.interpolate(imgs, size=self.resize_px))
        imgs = imgs.reshape((num_batches, -1) + (imgs.shape[-3:]))
        if self.sampler.freeze_all:
            probs = probs.detach()

        selected_imgs, distribution, policy, sample_index, sample_probs = self.sample(imgs, probs, test_mode)
        bs_imgs, bs_distribution, bs_policy, bs_sample_index, bs_sample_probs = self.sample(imgs, probs, test_mode, self.bp_mode)

        return selected_imgs, bs_imgs, distribution, policy, sample_index, sample_probs

    def forward_train(self, imgs, labels, **kwargs):
        num_batches = imgs.shape[0]

        if hasattr(self, 'sampler'):
            imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))

            if self.sampler.freeze_all:
                imgs, bs_imgs, distribution, policy, sample_index, sample_probs = self.forward_sampler(imgs, num_batches, True)
            else:
                imgs, bs_imgs, distribution, policy, sample_index, sample_probs = self.forward_sampler(imgs, num_batches, **kwargs)


        imgs = imgs.transpose(1, 2).contiguous()
        bs_imgs = bs_imgs.transpose(1, 2).contiguous()

        losses = dict()
        x = self.extract_feat(imgs)
        bs_x = self.extract_feat(bs_imgs)

        cls_score = self.cls_head(x)
        bs_cls_score = self.cls_head(bs_x)

        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        if gt_labels.shape == torch.Size([]):
            gt_labels = gt_labels.unsqueeze(0)

        reward_list = []
        origin_reward_list = []
        bs_reward_list = []
        for i in range(gt_labels.shape[0]):
            gt_label = gt_labels[i]

            gt_score = F.softmax(cls_score[i], dim=0)[gt_label].unsqueeze(0)
            max_score, max_index = torch.max(F.softmax(cls_score[i], dim=0), 0)
            if max_index == gt_label:
                reward = gt_score
            else:
                reward = gt_score - max_score

            bs_gt_score = F.softmax(bs_cls_score[i], dim=0)[gt_label].unsqueeze(0)
            bs_max_score, bs_max_index = torch.max(F.softmax(bs_cls_score[i], dim=0), 0)
            if bs_max_index == gt_label:
                bs_reward = bs_gt_score
            else:
                bs_reward = bs_gt_score - bs_max_score
            bs_reward = bs_reward.detach()
            advantage = reward - bs_reward

            origin_reward_list.append(reward)
            bs_reward_list.append(bs_reward)
            reward_list.append(advantage)
        reward = torch.cat(reward_list).clone().detach()
        origin_reward = torch.cat(origin_reward_list).clone().detach()
        bs_reward = torch.cat(bs_reward_list).clone().detach()

        loss_cls['reward'] = reward.mean()
        loss_cls['origin_reward'] = origin_reward.mean()
        loss_cls['bs_reward'] = bs_reward.mean()

        if self.sampler.freeze_all:
            loss_cls['all_probs'] = sample_probs.clone().detach().mean()
            loss_cls['loss_cls'] = loss_cls['loss_cls']
            losses.update(loss_cls)
            return losses

        entropy = torch.sum(-distribution * torch.log(distribution), dim=1)

        sample_probs = sample_probs.clamp(1e-15, 1-1e-15)

        if self.calc_mode == 'approximate':
            eps = 1e-9
            sample_probs, _ = torch.sort(sample_probs, dim=1)
            sample_probs = sample_probs[:, :min(self.num_segments, 8)]

            ones = torch.ones(sample_probs.shape[0], sample_probs.shape[1], device=imgs.device)
            sample_probs_reverse = torch.flip(sample_probs, [-1])

            divisor = ones - F.pad(sample_probs.cumsum(dim=1)[:, :-1], (1, 0))
            divisor_reverse = ones - F.pad(sample_probs_reverse.cumsum(dim=1)[:, :-1], (1, 0))

            sample_probs_1 = torch.cumprod(sample_probs / divisor, dim=1)[:, -1]
            sample_probs_2 = torch.cumprod(sample_probs_reverse / divisor_reverse, dim=1)[:, -1]

            sample_probs = (sample_probs_1 + sample_probs_2) / 2 * math.factorial(min(self.num_segments, 8))
            policy_cross_entropy = torch.log(sample_probs)
        elif self.calc_mode == 'all':
            dividend = torch.cumprod(sample_probs, dim=1)[:, -1]
            permute_index = torch.tensor(self.permute_index, dtype=torch.long, device=imgs.device)
            multi_index = permute_index.repeat(1, num_batches).reshape(self.index_length * num_batches, self.num_segments)
            multi_probs = sample_probs.repeat(self.index_length, 1)
            batch_inds = torch.arange(self.index_length * num_batches, device=imgs.device).unsqueeze(-1).expand_as(multi_index)
            multi_probs = multi_probs[batch_inds, multi_index]
            multi_probs = multi_probs.reshape(-1, num_batches, self.num_segments)
            ones = torch.ones(self.index_length, num_batches, self.num_segments, device=imgs.device)
            divisor = ones - F.pad(multi_probs.cumsum(dim=2)[:, :, :-1], (1, 0))
            divisor = torch.cumprod(divisor, dim=2)[:, :, -1]
            sample_probs = (dividend / divisor).sum(dim=0)

            policy_cross_entropy = torch.log(sample_probs)

        if self.cls_head.final_loss:
            loss_cls_ = -(reward * policy_cross_entropy).mean()
            loss_cls['entropy_fc'] = loss_cls['loss_cls'].clone().detach()
            loss_cls['sampler_loss'] = loss_cls_.clone().detach()
            loss_cls['loss_cls'] = loss_cls_ + loss_cls['loss_cls']
            loss_cls['all_probs'] = sample_probs.clone().detach().mean()
        else:
            loss_cls_ = -(reward * policy_cross_entropy + self.explore_rate * entropy).mean()
            loss_cls['loss_cls'] = loss_cls_
            loss_cls['entropy'] = entropy.clone().detach() * self.explore_rate
            loss_cls['explore_rate'] = torch.tensor([self.explore_rate], device=imgs.device)
            loss_cls['all_probs'] = sample_probs.clone().detach().mean()

        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs, **kwargs):
        num_batches = imgs.shape[0]
        num_clips = imgs.shape[1] // self.sampler.total_segments

        imgs = imgs.reshape((-1,) + (imgs.shape[-3:]))

        if self.max_testing_views is not None:
            print('Here?')
            total_views = imgs.shape[0]
            view_ptr = 0
            cls_scores = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                batch_imgs, _, distribution, _, sample_index, _ = self.forward_sampler(batch_imgs, self.max_testing_views//self.sampler.total_segments, test_mode=True)
                batch_imgs = batch_imgs.transpose(1, 2).contiguous()
                x = self.extract_feat(batch_imgs)
                cls_score = self.cls_head(x)
                cls_scores.append(cls_score)
                view_ptr += self.max_testing_views
            cls_score = torch.cat(cls_scores)
        else:# img B T C H W
            imgs, _, distribution, _, sample_index, _ = self.forward_sampler(imgs, num_batches * num_clips, test_mode=True)
            imgs = imgs.transpose(1, 2).contiguous()
            # print(f'in sampler, imgs shape: {imgs.shape}')
            x = self.extract_feat(imgs) # B C T H W
            # print(f'in sampler, feats shape: {x.shape}')
            cls_score = self.cls_head(x)
            # print(f'in sampler, after cls head shape: {cls_score.shape}')
        cls_score = self.average_clip(cls_score, num_clips)
        # print(f'in sampler, after avg clip shape: {cls_score.shape}')
        # raise NotImplementedError
        return cls_score.cpu().numpy()

    def forward_test(self, imgs, **kwargs):
        return self._do_test(imgs, **kwargs)

    def forward_gradcam(self, imgs):
        pass

@RECOGNIZERS.register_module()
class GumbelSampler2DRecognizer3D(BaseRecognizer):
    """Gumbel 2D Sampler 2D recognizer model framework."""
    def __init__(self,
                 backbone,
                 cls_head,
                 sampler=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_sampler=False,
                 loss=None,
                 loss_lambda=2.0,
                 loss_correction=False,
                 ce_lambda=0.0,
                 broken=True,
                 nout=1,
                 resize_px=None):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        
        self.resize_px=resize_px
        self.loss=loss
        self.loss_correction=loss_correction
        self.broken=broken
        assert self.loss in ['tan', 'smoothl1', 'log', 'exp', 'exp_tan', 'power', None]
        self.loss_lambda=loss_lambda
        self.ce_lambda=ce_lambda
        self.nout=nout
    
    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        
        if self.resize_px is None:
            probs = self.sampler(imgs)
        else:
            probs = self.sampler(F.interpolate(imgs, size=self.resize_px))
            assert imgs.shape[-1] == 224
        imgs = imgs.reshape((batches, -1) + (imgs.shape[-3:]))
        
        reverse_probs = (1 - probs).softmax(-1)
        max_index = probs.max(-1, keepdim=True)[1]
        min_index = reverse_probs.max(-1, keepdim=True)[1]
        pmax = torch.zeros_like(probs, memory_format=torch.legacy_contiguous_format).scatter_(-1, max_index, 1.0)
        pmin = torch.zeros_like(reverse_probs, memory_format=torch.legacy_contiguous_format).scatter_(-1, min_index, 1.0)
        pmax = pmax - probs.detach() + probs
        pmin = pmin - reverse_probs.detach() + reverse_probs
        
        # sampled_imgs = imgs - imgs * probs[:,:,None,None,None]
        if self.broken:
            sampled_imgs = imgs - imgs * pmax[:,:,None,None,None] + (imgs * pmin[:,:,None,None,None]).sum(1, keepdim=True) * pmax[:,:,None,None,None]
        else:
            sampled_imgs = imgs - imgs * pmax[:,:,None,None,None]
            min_indice = reverse_probs.topk(reverse_probs.shape[1]-1, dim=1)[1]
            num_batches = min_indice.shape[0]
            batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(min_indice)
            sampled_imgs = sampled_imgs[batch_inds, min_indice]
            
        # opp_probs = torch.ones_like(probs).to(probs.device) - probs
        # imgs = imgs * opp_probs[:, :, None, None, None]
        # B, T, C, H, W -> B, C, T, H, W  
        imgs = imgs.transpose(1, 2).contiguous()
        sampled_imgs = sampled_imgs.transpose(1, 2).contiguous()
        losses = dict()
        
        ori_x = self.extract_feat(imgs)
        ori_cls_score = self.cls_head(ori_x)
        
        x = self.extract_feat(sampled_imgs)
        cls_score = self.cls_head(x)
        
        gt_labels = labels.squeeze()
        label_cls_score = cls_score[range(cls_score.shape[0]),gt_labels]
        label_ori_cls_score = ori_cls_score[range(cls_score.shape[0]),gt_labels]
        margin = label_cls_score - label_ori_cls_score

        if self.loss_correction:
            positive = margin >= 0
            negative = margin < 0
            margin[positive] = margin[positive] / (1-label_ori_cls_score)[positive]
            margin[negative] = margin[negative] / (label_ori_cls_score)[negative]
        if self.loss == 'tan':
            losses['tan_loss'] = (- self.loss_lambda * torch.tan(np.pi/2*margin)).mean()
        elif self.loss == 'smoothl1':
            # l1_loss = torch.nn.SmoothL1Loss(beta=2.0)
            # loss = self.loss_lambda * l1_loss(ori_cls_score[:,labels],cls_score[:,labels])
            losses['smoothl1_loss'] = (self.loss_lambda * SmoothedL1HingeLoss(margin)).mean()
        elif self.loss == 'log':
            losses['log_loss'] = (- self.loss_lambda * torch.log(margin+1+1e-10)).mean()
        elif self.loss == 'exp':
            losses['exp_loss'] = (self.loss_lambda * torch.exp(-margin)).mean()
        elif self.loss == 'exp_tan':
            losses['exp_tan_loss'] = torch.exp((- self.loss_lambda * torch.tan(np.pi/2*margin))).mean()
        elif self.loss == 'power':
            losses['power_loss'] = (margin - 1) ** self.loss_lambda
        else:
            losses = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        if self.loss and self.ce_lambda:
            losses['loss_cls'] =  self.ce_lambda * self.cls_head.loss(cls_score, gt_labels, **kwargs)['loss_cls']
        top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                           labels.detach().cpu().numpy(), (1, 5))
        losses['top1_acc'] = torch.tensor(
            top_k_acc[0], device=cls_score.device)
        losses['top5_acc'] = torch.tensor(
            top_k_acc[1], device=cls_score.device)
        # losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        
        batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.sampler.is_shift:
            for _ in range(self.nout):
                num_segs = imgs.shape[0] // batches
                if self.resize_px is None:
                    probs = self.sampler(imgs, num_segs)
                else:
                    probs = self.sampler(F.interpolate(imgs, size=self.resize_px), num_segs)
                    assert imgs.shape[-1] == 224
                
                reverse_probs = (1 - probs).softmax(-1)
                max_index = probs.max(-1, keepdim=True)[1]
                pmax = torch.zeros_like(probs, memory_format=torch.legacy_contiguous_format).scatter_(-1, max_index, 1.0)

                min_index = reverse_probs.max(-1, keepdim=True)[1]
                min_indice = reverse_probs.topk(reverse_probs.shape[1]-1, dim=1)[1]
                pmin = torch.zeros_like(reverse_probs, memory_format=torch.legacy_contiguous_format).scatter_(-1, min_index, 1.0)
                imgs = imgs.reshape((batches, -1) + (imgs.shape[-3:]))
                if self.broken:
                    imgs = imgs - imgs * pmax[:,:,None,None,None] + (imgs * pmin[:,:,None,None,None]).sum(1, keepdim=True) * pmax[:,:,None,None,None] #
                else:
                    num_batches = min_indice.shape[0]
                    batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(min_indice)
                    imgs = imgs[batch_inds, min_indice]
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        else:
            ret = self.sampler._inner_forward(x)
            for _ in range(self.nout):
                num_segs = ret.shape[0] // batches
                if self.resize_px is None:
                    probs = self.sampler._sampling_forward(ret, num_segs)
                else:
                    probs = self.sampler(F.interpolate(ret, size=self.resize_px), num_segs)
                    assert imgs.shape[-1] == 224
                
                reverse_probs = (1 - probs).softmax(-1)
                max_index = probs.max(-1, keepdim=True)[1]
                pmax = torch.zeros_like(probs, memory_format=torch.legacy_contiguous_format).scatter_(-1, max_index, 1.0)

                min_index = reverse_probs.max(-1, keepdim=True)[1]
                min_indice = reverse_probs.topk(reverse_probs.shape[1]-1, dim=1)[1]
                pmin = torch.zeros_like(reverse_probs, memory_format=torch.legacy_contiguous_format).scatter_(-1, min_index, 1.0)
                imgs = imgs.reshape((batches, -1) + (imgs.shape[-3:]))
                ret = ret.reshape((batches, -1) + (ret.shape[-3:]))
                if self.broken:
                    imgs = imgs - imgs * pmax[:,:,None,None,None] + (imgs * pmin[:,:,None,None,None]).sum(1, keepdim=True) * pmax[:,:,None,None,None] #
                else:
                    num_batches = min_indice.shape[0]
                    batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(min_indice)
                    imgs = imgs[batch_inds, min_indice]
                    ret = ret[batch_inds, min_indice]
                ret = ret.reshape((-1, ) + ret.shape[2:])

        sampled_imgs = imgs
        x = self.extract_feat(sampled_imgs)
        
        num_segs = imgs.shape[0] // batches
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

        outs = (self.cls_head(x, num_segs), )
        return outs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)
