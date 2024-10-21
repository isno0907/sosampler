import torch
import torch.nn.functional as F

import numpy as np

from ..registry import RECOGNIZERS
from .base import BaseRecognizer
from ...core import top_k_accuracy, top_k_recall
from itertools import permutations

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
class Sampler2DRecognizer2D(BaseRecognizer):

    def __init__(self,
                 sampler,
                 backbone,
                 cls_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 bp_mode='gradient_policy',
                 calc_mode='all',
                 reward_kind=1,
                 num_segments=4,
                 resize_px=None,
                 num_test_segments=None,
                 use_sampler=False,
                 explore_rate=1.,
                 clamp_range=None,
                 num_clips=1,
                 cal_number=None):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        self.cal_number = cal_number
        self.clamp_range = clamp_range
        self.resize_px = resize_px
        self.num_segments = num_segments
        self.bp_mode = bp_mode
        self.num_clips = num_clips
        self.calc_mode = calc_mode
        self.reward_kind = reward_kind
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
            # print(probs.shape)
            num_batches, original_segments = probs.shape
            sample_index = probs.topk(self.num_test_segments, dim=1)[1]
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
        if self.resize_px is None:
            probs = self.sampler(imgs)
        else:
            probs = self.sampler(F.interpolate(imgs, size=self.resize_px))
            # assert imgs.shape[-1] == 224
        imgs = imgs.reshape((num_batches, -1) + (imgs.shape[-3:]))

        selected_imgs, distribution, policy, sample_index, sample_probs = self.sample(imgs, probs, test_mode)
        bs_imgs, bs_distribution, bs_policy, bs_sample_index, bs_sample_probs = self.sample(imgs, probs, test_mode,
                                                                                            self.bp_mode)

        return selected_imgs, bs_imgs, distribution, policy, sample_index, sample_probs

    def forward_train(self, imgs, labels, **kwargs):
        num_batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
        imgs, bs_imgs, distribution, policy, sample_index, sample_probs = self.forward_sampler(imgs, num_batches, **kwargs)

        losses = dict()
        num_segs = imgs.shape[1]

        imgs = imgs.reshape((-1, ) + imgs.shape[-3:])
        bs_imgs = bs_imgs.reshape((-1, ) + imgs.shape[-3:])
        x = self.extract_feat(imgs)
        bs_x = self.extract_feat(bs_imgs)

        cls_score = self.cls_head(x, num_segs)
        bs_cls_score = self.cls_head(bs_x, num_segs)

        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        if gt_labels.shape == torch.Size([]):
            gt_labels = gt_labels.unsqueeze(0)

        if self.reward_kind in [2, 3]:
            avg_x = self.cls_head.avg_pool_2d(x)
            avg_x = avg_x.reshape(-1, self.cls_head.in_channels)
            if hasattr(self.cls_head, 'layers'):
                # for frameExit checkpoint
                avg_x = self.cls_head.layers(avg_x)
                avg_x = avg_x.reshape(-1, num_segs, self.cls_head.num_neurons[-1])
                if hasattr(self.cls_head, 'fc'):
                    cls_score_seg = self.cls_head.fc(avg_x)
                else:
                    cls_score_seg = self.cls_head.fc_list[0](avg_x)
            else:
                cls_score_seg = self.cls_head.fc_cls(avg_x)
                cls_score_seg = cls_score_seg.reshape(num_batches, num_segs, -1)
            cls_score_seg = F.softmax(cls_score_seg, dim=2)
            cls_score_seg = cls_score_seg.detach()
            # [N, seg, num_classes]

            bs_avg_x = self.cls_head.avg_pool_2d(bs_x)
            bs_avg_x = bs_avg_x.reshape(-1, self.cls_head.in_channels)
            if hasattr(self.cls_head, 'layers'):
                bs_avg_x = self.cls_head.layers(bs_avg_x)
                bs_avg_x = bs_avg_x.reshape(-1, num_segs, self.cls_head.num_neurons[-1])
                if hasattr(self.cls_head, 'fc'):
                    bs_cls_score_seg = self.cls_head.fc(bs_avg_x)
                else:
                    bs_cls_score_seg = self.cls_head.fc_list[0](bs_avg_x)
            else:
                bs_cls_score_seg = self.cls_head.fc_cls(bs_avg_x)
                bs_cls_score_seg = bs_cls_score_seg.reshape(num_batches, num_segs, -1)
            bs_cls_score_seg = F.softmax(bs_cls_score_seg, dim=2)
            bs_cls_score_seg = bs_cls_score_seg.detach()

        reward_list = []
        origin_reward_list = []
        bs_reward_list = []
        for i in range(gt_labels.shape[0]):
            gt_label = gt_labels[i]
            if self.reward_kind == 1:
                max_score_lb, max_index_lb = torch.max(cls_score[i], 0)
                bs_max_score_lb, bs_max_index_lb = torch.max(bs_cls_score[i], 0)
                origin_reward = torch.cuda.FloatTensor([max_score_lb], device=cls_score.device).detach()
                origin_reward_list.append(origin_reward)
                bs_reward = torch.cuda.FloatTensor([bs_max_score_lb], device=cls_score.device).detach()
                bs_reward_list.append(bs_reward)
                reward = max_score_lb - bs_max_score_lb
                reward = torch.cuda.FloatTensor([reward], device=cls_score.device).detach()
                reward_list.append(reward)
                continue

            if self.reward_kind == 2:
                reward = 0
                seg_scores = cls_score_seg[i]
                max_scores, max_index = torch.max(seg_scores, 1)
                for j, seg_score in enumerate(seg_scores):
                    if max_index[j] == gt_label:
                        reward += max_scores[j].unsqueeze(0)
                    else:
                        reward += seg_score[gt_label].unsqueeze(0) - max_scores[j].unsqueeze(0)
                reward = reward / self.num_segments
                origin_reward_list.append(reward)

                bs_reward = 0
                bs_seg_scores = bs_cls_score_seg[i]
                bs_max_scores, bs_max_index = torch.max(bs_seg_scores, 1)
                for j, bs_seg_score in enumerate(bs_seg_scores):
                    if bs_max_index[j] == gt_label:
                        bs_reward += bs_max_scores[j].unsqueeze(0)
                    else:
                        bs_reward += bs_seg_score[gt_label].unsqueeze(0) - bs_max_scores[j].unsqueeze(0)
                bs_reward = bs_reward / self.num_segments
                bs_reward_list.append(bs_reward)

                reward_list.append(reward - bs_reward)
                continue

            if self.reward_kind == 3:
                seg_scores = cls_score_seg[i]
                seg_scores = seg_scores[:, gt_label]
                reward = torch.max(seg_scores)
                reward = reward.unsqueeze(0)
                reward_list.append(reward)
                continue
            raise RuntimeError(f"reward kind {self.reward_kind} is not support")

        reward = torch.cat(reward_list).clone().detach()
        loss_cls['reward'] = reward.mean()
        if origin_reward_list != []:
            origin_reward = torch.cat(origin_reward_list).clone().detach()
            loss_cls['origin_reward'] = origin_reward.mean()
        if bs_reward_list != []:
            bs_reward = torch.cat(bs_reward_list).clone().detach()
            loss_cls['bs_reward'] = bs_reward.mean()

        entropy = torch.sum(-distribution * torch.log(distribution), dim=1)

        if self.clamp_range is None:
            sample_probs = sample_probs.clamp(1e-15, 1-1e-15)

        if self.calc_mode == 'approximate':
            eps = 1e-9
            ones = torch.ones(sample_probs.shape[0], sample_probs.shape[1], device=imgs.device)
            divisor = ones - F.pad(sample_probs.cumsum(dim=1)[:, :-1], (1,0))

            sample_probs = sample_probs / divisor
            policy_cross_entropy = torch.sum(torch.log(sample_probs + eps), dim=1)
        elif self.calc_mode == 'all':
            if self.cal_number is not None:
                origin_prob = sample_probs.clone().detach()
                sample_probs = sample_probs[:, :self.cal_number]
                sample_probs = F.softmax(sample_probs, dim=1)
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
            if self.calc_mode == 'all':
                loss_cls['all_probs'] = sample_probs.clone().detach().mean()
        else:
            loss_cls_ = -(reward * policy_cross_entropy + self.explore_rate * entropy).mean()
            loss_cls['loss_cls'] = loss_cls_
            loss_cls['entropy'] = entropy.clone().detach() * self.explore_rate
            loss_cls['explore_rate'] = torch.tensor([self.explore_rate], device=imgs.device)
            if self.calc_mode == 'all':
                if (self.cal_number is not None) or (self.clamp_range is not None):
                    loss_cls['all_probs'] = origin_prob.mean()
                else:
                    loss_cls['all_probs'] = sample_probs.clone().detach().mean()

        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs, **kwargs):
        num_batches = imgs.shape[0]

        imgs = imgs.reshape((-1, ) + (imgs.shape[-3:]))
        imgs, _, distribution, _, sample_index, _ = self.forward_sampler(imgs, num_batches * self.num_clips, test_mode=True)
        num_clips_crops = imgs.shape[0] // num_batches
        num_segs = imgs.shape[1]

        imgs = imgs.reshape((-1, ) + imgs.shape[-3:])
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x, num_segs)
        cls_score = self.average_clip(cls_score, num_clips_crops)
        return cls_score.cpu().numpy()

    def forward_test(self, imgs, **kwargs):
        return self._do_test(imgs, **kwargs)

    def forward_gradcam(self, imgs):
        pass

@RECOGNIZERS.register_module()
class GumbelSampler2DRecognizer2D(BaseRecognizer):
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
                 random_nseg=False,
                 nlow=6,
                 resize_px=None):
        super().__init__(backbone, cls_head, sampler, neck, train_cfg, test_cfg, use_sampler)
        
        self.resize_px=resize_px
        self.loss=loss
        self.loss_correction=loss_correction
        self.broken=broken
        # assert self.loss in ['tan', 'smoothl1', 'log', 'exp', 'exp_tan', 'power', None]
        self.loss_lambda=loss_lambda
        self.ce_lambda=ce_lambda
        self.nout=nout
        self.random_nseg=random_nseg
        self.nlow=nlow
    
    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        batches, num_segs = imgs.shape[:2]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        
        if self.resize_px is None:
            probs = self.sampler(imgs, num_segs)
        else:
            probs = self.sampler(F.interpolate(imgs, size=self.resize_px), num_segs)
            assert imgs.shape[-1] == 224
        
        imgs = imgs.reshape((batches, -1) + (imgs.shape[-3:]))

        # sampled_imgs = imgs - imgs * pmax[:,:,None,None,None] + (imgs * pmin[:,:,None,None,None]).sum(1, keepdim=True) * pmax[:,:,None,None,None]
        imgs = (imgs * probs[:,:,None,None,None]).sum(1, keepdim=True)
        imgs = imgs.reshape((-1, ) + imgs.shape[-3:])

        # print(imgs.shape)
        
        losses = dict()
        
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x, 1)
        
        gt_labels = labels.squeeze()
        print(gt_labels.shape)
        print(gt_labels)
        losses = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches, num_segs = imgs.shape[:2]

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        if self.resize_px is None:
            probs = self.sampler(imgs, num_segs)
        else:
            probs = self.sampler(F.interpolate(imgs, size=self.resize_px), num_segs)
            assert imgs.shape[-1] == 224
        
        imgs = imgs.reshape((batches, -1) + (imgs.shape[-3:]))
        # imgs = imgs[torch.arange(batches),probs.argmax(-1)]
        
        n=6
        num_batches, original_segments = probs.shape
        sample_index = probs.topk(n, dim=1)[1]
        sample_index, _ = sample_index.sort(dim=1, descending=False)
        batch_inds = torch.arange(num_batches).unsqueeze(-1).expand_as(sample_index)

        imgs = imgs[batch_inds, sample_index]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # print(f"{imgs.shape}")

        # sampled_imgs = imgs - imgs * pmax[:,:,None,None,None] + (imgs * pmin[:,:,None,None,None]).sum(1, keepdim=True) * pmax[:,:,None,None,None]
        # imgs = (imgs * probs[:,:,None,None,None]).sum(1, keepdim=True)

        # print(imgs.shape)
        losses = dict()
        
        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x, n)


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
