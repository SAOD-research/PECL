import logging
import os.path as osp
from shlex import join
import time
from tkinter import W
from unittest import result

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import cv2
import copy
import mmcv
from . import hooks
from .checkpoint import load_checkpoint, save_checkpoint
from .hooks import (CheckpointHook, Hook, IterTimerHook, LrUpdaterHook,
                    OptimizerHook, lr_updater)
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import get_dist_info, get_host_info, get_time_str, obj_from_dict
import pdb
import numpy as np
from .transforms import BboxTransform, MaskTransform
from .DDPG import DDPG, ReplayBuffer
from pycocotools import mask as maskUtils
# from .. import (ImageTransform, BboxTransform, MaskTransform,
#                          SegMapTransform, Numpy2Tensor)

from .transforms_rbbox import gt_mask_bp_obbs_list, choose_best_Rroi_batch, dbbox2roi
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
               'helicopter']
classthreshold = torch.tensor([0, 0.6, 0.7, 0.8, 0.8, 0.5, 0.4, 0.4, 0.6, 0.7, 0.6, 0.7, 0.88, 0.6, 0.6, 0.6])


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    # Intersection area 这里没有对inter_area为负的情况进行判断，后面计算出来的IOU就可能是负的
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def result2bboxmask(result, img_shape, scale_factor, pad_shape, flip):
    # result=torch.trunc(result)
    xmin, ymin, xmax, ymax = result[:, :8:2].min(1)[0], result[:, 1::2].min(1)[0], result[:, :8:2].max(1)[0], \
    result[:, 1::2].max(1)[0]
    # width, height = xmax - xmin, ymax - ymin
    gt_bboxes = torch.t(torch.cat(
        (xmin.reshape(1, -1), ymin.reshape(1, -1), (xmax - 1).reshape(1, -1), (ymax - 1).reshape(1, -1)))).type(
        torch.float32)
    gt_bboxes = torch.trunc(gt_bboxes)
    # pdb.set_trace()
    gt_masks = []
    for ann in result[:, 0:8]:
        # pdb.set_trace()
        rles = maskUtils.frPyObjects([ann.cpu().tolist()], 1024, 1024)
        rle = maskUtils.merge(rles)
        gt_mask = maskUtils.decode(rle)
        # if len(gt_mask[gt_mask>0])==0:
        #     pdb.set_trace()
        gt_masks.append(gt_mask)
    if len(gt_bboxes) == 0:
        pdb.set_trace()
        return None
    bbox_transform = BboxTransform()
    mask_transform = MaskTransform()
    gt_bboxes = bbox_transform(gt_bboxes.cpu().numpy(), img_shape, scale_factor, flip)
    gt_masks = mask_transform(gt_masks, pad_shape, scale_factor, flip)
    # pdb.set_trace()
    return gt_bboxes, gt_masks


import math


def entropy(Plist):
    if len(Plist):
        result = 0
        for x in Plist:
            result += (-x) * math.log(x, 2)
    else:
        print('Error loading data')
        pdb.set_trace()
        result = -1
    return result


class TensorboardSummary(object):
    def __init__(self, directory, use_dist=False):
        self.directory = directory
        self.use_dist = use_dist

    def create_summary(self):
        writer = SummaryWriter(logdir=os.path.join(self.directory))
        return writer

input_dim = 8
num_action = 2
buffer_size = 1000
min_size = 200
batch_size = 64

args = {'actor_optimizer': 'sgd',
        'critic_optimizer': 'sgd',
        'actor_lr': 1e-3,
        'critic_lr': 5e-4,
        'rl_sui': 1,
        'rl_momentum': 0.01,
        'rl_resume': None,
        'rl_optim_interval': 1,
        'rl_ft': False,
        'rl_discount': 0.9,
        'cuda': True}


class Runner(object):
    """A training helper for PyTorch.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None):
        assert callable(batch_processor)

        self.ddpg = DDPG(args, input_dim, num_action)
        self.rl_update_step = 0
        # 经验回放池实例化
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.batch_processor = batch_processor

        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(optimizer, torch.optim,
                                      dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = '{}.log'.format(self.timestamp)
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # use relative symlink
        mmcv.symlink(filename, linkpath)

    def train(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'train'
        self.ddpg.train()
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')

        self.summary = TensorboardSummary(
            "/workspace/ReDet/work_dirs/ReDet_re50_refpn_1x_dota1_0.01ddpgp/epoch{}".format(self._epoch + 1))
        self.writer = self.summary.create_summary()

        for i, data_batch in enumerate(data_loader):
            gt_bboxes = data_batch['gt_bboxes'].data[0]
            gt_labels = data_batch['gt_labels'].data[0]
            gt_masks = data_batch['gt_masks'].data[0]

            data_batch1 = {}
            data_batch2 = {}
            for key, value in data_batch.items():
                if key == 'img_meta':
                    data_batch1[key] = [[value.data[0][0]]]
                    data_batch2[key] = [[value.data[0][1]]]
                elif key == 'img':
                    data_batch1[key] = [value.data[0][[True, False]]]
                    data_batch2[key] = [value.data[0][[False, True]]]

            result1box = []
            result1label = []
            fullname = os.path.join('/workspace/ReDet/datasets/DOTA_1024/trainval1024/pselabelTxt_0.01',
                                    data_batch['img_meta'].data[0][0]['name'].strip('.png') + '.txt')
            f = open(fullname, 'r')
            for line in f.readlines():
                splitlines = line.strip().split(' ')
                result1box.append(
                    [float(splitlines[0]), float(splitlines[1]), float(splitlines[2]), float(splitlines[3]),
                     float(splitlines[4]), float(splitlines[5]), float(splitlines[6]), float(splitlines[7]),
                     float(splitlines[8])])
                result1label.append(int(splitlines[9]))
            result1box = torch.from_numpy(np.array(result1box)).to('cuda')
            result1label = torch.from_numpy(np.array(result1label)).to('cuda')

            result2box = []
            result2label = []
            fullname = os.path.join('/workspace/ReDet/datasets/DOTA_1024/trainval1024/pselabelTxt_0.01',
                                    data_batch['img_meta'].data[0][1]['name'].strip('.png') + '.txt')
            f = open(fullname, 'r')
            for line in f.readlines():
                splitlines = line.strip().split(' ')
                result2box.append(
                    [float(splitlines[0]), float(splitlines[1]), float(splitlines[2]), float(splitlines[3]),
                     float(splitlines[4]), float(splitlines[5]), float(splitlines[6]), float(splitlines[7]),
                     float(splitlines[8])])
                result2label.append(int(splitlines[9]))
            result2box = torch.from_numpy(np.array(result2box)).to('cuda')
            result2label = torch.from_numpy(np.array(result2label)).to('cuda')

            if len(result1box) != 0:
                # 4点水平框 mask
                result1gt_bboxes, result1gt_masks = result2bboxmask(result1box,
                                                                    data_batch1['img_meta'][0][0]['img_shape'],
                                                                    data_batch1['img_meta'][0][0]['scale_factor'],
                                                                    data_batch1['img_meta'][0][0]['pad_shape'],
                                                                    data_batch1['img_meta'][0][0]['flip'])

                # 伪标签与gt计算iou，筛去nms大的
                if len(result1label) > 0 and len(gt_bboxes[0]) > 0:
                    gt = torch.cat([gt_bboxes[0]] * len(result1label), dim=1).reshape(-1, 4)
                    pred = torch.cat([torch.from_numpy(result1gt_bboxes)] * len(gt_bboxes[0]), dim=0)
                    if len(pred) == 0:
                        pdb.set_trace()
                    iou = bbox_iou(gt, pred).reshape(len(gt_bboxes[0]), -1).to('cuda')
                    classcompare = torch.mm(torch.nn.functional.one_hot(gt_labels[0] - 1, 15).float().to('cuda'),
                                            torch.nn.functional.one_hot(result1label - 1, 15).float().t())
                    index = torch.where(iou * classcompare > 0.7)[1].cpu().tolist()
                    indexx = list(set([x for x in range(len(result1label))]) - set(index))
                    result1box = torch.index_select(result1box, 0, torch.tensor(indexx).to('cuda'))
                    result1label = \
                    torch.index_select(result1label.reshape((-1, 1)), 0, torch.tensor(indexx).to('cuda')).reshape(
                        (1, -1))[0]
                    result1gt_bboxes = torch.index_select(torch.from_numpy(result1gt_bboxes), 0,
                                                          torch.LongTensor(indexx))
                    result1gt_masks = result1gt_masks[indexx, :]

                # 筛选置信度低于阈值的
                # index1=torch.where(result1box[:,8]>0.1)[0].to('cpu').numpy()

                # result1box = torch.index_select(result1box, 0, torch.tensor(index1).to('cuda'))
                # result1label = torch.index_select(result1label.reshape((-1,1)), 0, torch.tensor(index1).to('cuda')).reshape((1,-1))[0]
                # result1gt_bboxes=torch.index_select(result1gt_bboxes, 0, torch.LongTensor(index1))
                # result1gt_masks=result1gt_masks[index1,:]

                # #筛选一定数量的伪标签
                # if len(result1label)>int(NUM_PSESTATE/2):#随机sample
                #     # N,C = result1box.shape
                #     # S = int(NUM_PSESTATE/2)
                #     # pdb.set_trace()
                #     # index = torch.LongTensor(random.sample(range(N), S))
                #     _,index = result1box[:,-1].topk(int(NUM_PSESTATE/2),dim=0, largest=True, sorted=False)
                #     result1box = torch.index_select(result1box, 0, index.to('cuda'))
                #     result1label = torch.index_select(result1label.reshape((-1,1)), 0, index.to('cuda')).reshape((1,-1))[0]
                #     result1gt_bboxes=torch.index_select(result1gt_bboxes, 0, index.to('cpu'))
                #     result1gt_masks=result1gt_masks[index.to('cpu'),:]
                # # else:#填充
                # #     # pdb.set_trace()
                # #     result1box=torch.cat((result1box,torch.zeros((50-len(result1label),9)).to('cuda')))
                # #     result1label=torch.cat((result1label,torch.zeros(50-len(result1label)).to('cuda')))
            # pdb.set_trace()
            if len(result2box) != 0:
                result2gt_bboxes, result2gt_masks = result2bboxmask(result2box,
                                                                    data_batch2['img_meta'][0][0]['img_shape'],
                                                                    data_batch2['img_meta'][0][0]['scale_factor'],
                                                                    data_batch2['img_meta'][0][0]['pad_shape'],
                                                                    data_batch2['img_meta'][0][0]['flip'])

                if len(result2label) > 0 and len(gt_bboxes[1]) > 0:
                    gt = torch.cat([gt_bboxes[1]] * len(result2label), dim=1).reshape(-1, 4)
                    pred = torch.cat([torch.from_numpy(result2gt_bboxes)] * len(gt_bboxes[1]), dim=0)
                    iou = bbox_iou(gt, pred).reshape(len(gt_bboxes[1]), -1).to('cuda')
                    classcompare = torch.mm(torch.nn.functional.one_hot(gt_labels[1] - 1, 15).float().to('cuda'),
                                            torch.nn.functional.one_hot(result2label - 1, 15).float().t())
                    index = torch.where(iou * classcompare > 0.7)[1].cpu().tolist()
                    indexx = list(set([x for x in range(len(result2label))]) - set(index))
                    result2box = torch.index_select(result2box, 0, torch.tensor(indexx).to('cuda'))
                    result2label = \
                    torch.index_select(result2label.reshape((-1, 1)), 0, torch.tensor(indexx).to('cuda')).reshape(
                        (1, -1))[0]
                    result2gt_bboxes = torch.index_select(torch.from_numpy(result2gt_bboxes), 0,
                                                          torch.LongTensor(indexx))
                    result2gt_masks = result2gt_masks[indexx, :]

                # index2=torch.where(result2box[:,8]>0.1)[0].to('cpu').numpy()

                # result2box = torch.index_select(result2box, 0, torch.tensor(index2).to('cuda'))
                # result2label = torch.index_select(result2label.reshape((-1,1)), 0, torch.tensor(index2).to('cuda')).reshape((1,-1))[0]
                # result2gt_bboxes=torch.index_select(result2gt_bboxes, 0, torch.LongTensor(index2))
                # result2gt_masks=result2gt_masks[index2,:]

                # if len(result2label)>int(NUM_PSESTATE/2):#随机sample
                #     # N,C = result2box.shape
                #     # S = int(NUM_PSESTATE/2)
                #     # pdb.set_trace()
                #     # index = torch.LongTensor(random.sample(range(N), S))
                #     _,index = result2box[:,-1].topk(int(NUM_PSESTATE/2),dim=0, largest=True, sorted=False)
                #     result2box = torch.index_select(result2box, 0, index.to('cuda'))
                #     result2label = torch.index_select(result2label.reshape((-1,1)), 0, index.to('cuda')).reshape((1,-1))[0]
                #     result2gt_bboxes=torch.index_select(result2gt_bboxes, 0, index.to('cpu'))
                #     result2gt_masks=result2gt_masks[index.to('cpu'),:]
                # # else:#填充
                # #     # pdb.set_trace()
                # #     result2box=torch.cat((result2box,torch.zeros((50-len(result2label),9)).to('cuda')))
                # #     result2label=torch.cat((result2label,torch.zeros(50-len(result2label)).to('cuda')))

            # pdb.set_trace()

            # 概率、概率(背景概率)、相似度、共形值、软标签
            # self.rl_update_step = 0
            if len(result1label) != 0:
                state_pre1 = result1box[:, 8].reshape(-1, 1)
                state_plabel = result1label.reshape(-1, 1)
                state_gtpre1 = torch.tensor([1] * len(gt_labels[0])).cuda().reshape(-1, 1)
                state_gtplabel = gt_labels[0].cuda().reshape(-1, 1)

                pseudo_masks = [result1gt_masks]

                with torch.no_grad():
                    x = self.model.module.extract_feat(data_batch['img'].data[0].to('cuda'))

                    gt_obbs = gt_mask_bp_obbs_list(gt_masks)  # x,y,w,h,theta
                    gt_rrois = dbbox2roi(
                        [torch.from_numpy(choose_best_Rroi_batch(gt_obbs[0])).to('cuda').type(torch.float32)])
                    gt_rrois[:, 3] = gt_rrois[:, 3] * 1.2
                    gt_rrois[:, 4] = gt_rrois[:, 4] * 1.4
                    gt_rbbox_feats = self.model.module.rbbox_roi_extractor(x[:4],
                                                                           gt_rrois)  # torch.Size([6, 256, 7, 7])
                    if self.model.module.rbbox_head.num_shared_convs > 0:
                        for conv in self.model.module.rbbox_head.shared_convs:
                            gt_rbbox_feats = conv(gt_rbbox_feats)
                    if self.model.module.rbbox_head.num_shared_fcs > 0:
                        if self.model.module.rbbox_head.with_avg_pool:
                            gt_rbbox_feats = self.model.module.rbbox_head.avg_pool(gt_rbbox_feats)
                        gt_rbbox_feats = gt_rbbox_feats.view(gt_rbbox_feats.size(0), -1)
                        for fc in self.model.module.rbbox_head.shared_fcs:
                            gt_rbbox_feats = self.model.module.rbbox_head.relu(
                                fc(gt_rbbox_feats))  # torch.Size([6, 1024])
                    gt_rbbox_feats = F.layer_norm(gt_rbbox_feats, (1024,))
                    gt_rbbox_feats = F.normalize(gt_rbbox_feats, p=2, dim=-1)  # n,1024
                    gt_cls_score = self.model.module.rbbox_head.fc_cls(
                        gt_rbbox_feats) if self.model.module.rbbox_head.with_cls else None
                    gt_score = F.softmax(gt_cls_score * 10, dim=1) if gt_cls_score is not None else None
                    # 二次预测值
                    state_gtpre2 = gt_score.gather(1, gt_labels[0].to('cuda').reshape(-1, 1))

                    pseudo_obbs = gt_mask_bp_obbs_list(pseudo_masks)
                    pseudo_rrois = dbbox2roi(
                        [torch.from_numpy(choose_best_Rroi_batch(pseudo_obbs[0])).to('cuda').type(torch.float32)])
                    pseudo_rrois[:, 3] = pseudo_rrois[:, 3] * 1.2
                    pseudo_rrois[:, 4] = pseudo_rrois[:, 4] * 1.4
                    pseudo_rbbox_feats = self.model.module.rbbox_roi_extractor(x[:4],
                                                                               pseudo_rrois)  # torch.Size([6, 256, 7, 7])
                    if self.model.module.rbbox_head.num_shared_convs > 0:
                        for conv in self.model.module.rbbox_head.shared_convs:
                            pseudo_rbbox_feats = conv(pseudo_rbbox_feats)
                    if self.model.module.rbbox_head.num_shared_fcs > 0:
                        if self.model.module.rbbox_head.with_avg_pool:
                            pseudo_rbbox_feats = self.model.module.rbbox_head.avg_pool(pseudo_rbbox_feats)
                        pseudo_rbbox_feats = pseudo_rbbox_feats.view(pseudo_rbbox_feats.size(0), -1)
                        for fc in self.model.module.rbbox_head.shared_fcs:
                            pseudo_rbbox_feats = self.model.module.rbbox_head.relu(
                                fc(pseudo_rbbox_feats))  # torch.Size([6, 1024])
                    pseudo_rbbox_feats = F.layer_norm(pseudo_rbbox_feats, (1024,))
                    pseudo_rbbox_feats = F.normalize(pseudo_rbbox_feats, p=2, dim=-1)  # n,1024
                    pseudo_cls_score = self.model.module.rbbox_head.fc_cls(
                        pseudo_rbbox_feats) if self.model.module.rbbox_head.with_cls else None
                    pseudo_score = F.softmax(pseudo_cls_score * 10, dim=1) if pseudo_cls_score is not None else None
                    # 二次预测值
                    state_pre2 = pseudo_score.gather(1, result1label.to('cuda').reshape(-1, 1))

                    # class_dictionary = torch.load('prototypes.pt')#15,10,1024
                    class_dictionary = self.model.module.prototypes
                    # class_dictionary = F.normalize(class_dictionary, p=2, dim=-1)
                    # 相似度
                    class_dictionary_ave = torch.mean(class_dictionary, dim=1)
                    class_dictionary_ave = class_dictionary_ave / torch.norm(class_dictionary_ave, p=2, dim=1).reshape(
                        (-1, 1))
                    state_sim = torch.sum(
                        torch.mul(pseudo_rbbox_feats, class_dictionary_ave[(result1label - 1).tolist()]),
                        dim=1).reshape(-1, 1)
                    if gt_labels[0].shape[0] != gt_rbbox_feats.shape[0]:
                        pdb.set_trace()
                        print()
                    # pdb.set_trace()
                    state_gtsim = torch.sum(
                        torch.mul(gt_rbbox_feats, class_dictionary_ave[(gt_labels[0].to('cuda') - 1).tolist()]),
                        dim=1).reshape(-1, 1)

                    # 共形
                    sample_cpscore = 1 - state_pre2
                    sample_gtcpscore = 1 - state_gtpre2

                    calibration_cls_score = self.model.module.rbbox_head.fc_cls(
                        class_dictionary.reshape((-1, 1024))) if self.model.module.rbbox_head.with_cls else None
                    calibration_score = F.softmax(calibration_cls_score * 10,
                                                  dim=1) if calibration_cls_score is not None else None
                    calibration_pre = calibration_score.gather(1, torch.tensor(
                        [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10 + [6] * 10 + [7] * 10 + [8] * 10 + [
                            9] * 10 + [10] * 10 + [11] * 10 + [12] * 10 + [13] * 10 + [14] * 10 + [15] * 10).to(
                        'cuda').reshape(-1, 1))
                    calibration_cpscore = 1 - calibration_pre
                    calibration_cpscore = calibration_cpscore.reshape(15, 10)

                    temp = calibration_cpscore[(result1label - 1).tolist()]
                    state_cp = (((temp > sample_cpscore - 0.35) == True).sum(dim=1) / 10).reshape(-1, 1)
                    temp = calibration_cpscore[(gt_labels[0].to('cuda') - 1).tolist()]
                    state_gtcp = (((temp > sample_gtcpscore - 0.35) == True).sum(dim=1) / 10).reshape(-1, 1)

                    state_x = torch.cat((state_pre1, state_pre1, state_sim, state_cp), dim=1)
                    state_gt = torch.cat((state_gtpre1, state_gtpre1, state_gtsim, state_gtcp), dim=1)

                # 基于策略强化网络
                # 记录每个回合的return

                # 存放状态
                index_choose = []
                # 获取初始状态
                state = torch.cat((state_x[0], torch.mean(state_gt[:, 0:4], dim=0)))
                # 开始迭代
                for si in range(len(state_x) - 1):
                    # 动作选择
                    # action = self.agent.take_action(state)  # 对某一状态采取动作
                    action = self.ddpg._actor_forward(state)

                    classindex = state_plabel[si].item() - 1
                    choosedgt = torch.cat([calibration_score[classindex * 10:classindex * 10 + 10],
                                           gt_score[(state_gtplabel - 1).reshape(1, -1)[0] == classindex],
                                           pseudo_score[index_choose][
                                               (state_plabel[index_choose] - 1).reshape(1, -1)[0] == classindex]],
                                          dim=0)
                    vector1 = torch.mean(choosedgt, dim=0)
                    vector2 = torch.mean(torch.cat((choosedgt, pseudo_score[si].reshape(-1, 16)), dim=0), dim=0)

                    reward = entropy(vector2) - entropy(vector1) + (vector1[classindex + 1] - vector1[0]) - (
                                vector2[classindex + 1] - vector2[0]) - state_x[si][0].item() / 10 + 0.01
                    reward = reward.float()

                    if action == 1:
                        index_choose.append(si)

                    if action == 1:
                        reward = reward
                    else:
                        reward = -reward

                    if reward < 0:
                        reward = 1.0
                    else:
                        reward = -1.0

                    # 环境更新
                    next_state = torch.cat((state_x[si + 1],
                                            torch.mean(torch.cat((state_gt[:, 0:4], state_x[index_choose][:, :5 - 1])),
                                                       dim=0)), dim=0)

                    print(state_x[si], action, reward)
                    # 更新经验回放池
                    self.replay_buffer.add(state, torch.tensor([action]).cuda(), torch.tensor([reward]).cuda(),
                                           next_state)
                    # training_pair = [state, torch.tensor([action]).cuda(), torch.tensor([reward]).cuda(), next_state
                    state = next_state
                    # 如果经验池超过容量，开始训练
                    if self.replay_buffer.size() > min_size:
                        # 经验池随机采样batch_size组
                        s, a, r, ns = self.replay_buffer.sample(batch_size)
                        # 构造数据集
                        transition_dict = {
                            'states': s,
                            'actions': a,
                            'rewards': r,
                            'next_states': ns,
                        }
                        # 模型训练
                        value_loss, policy_loss = self.ddpg.update_policy(transition_dict, self.rl_update_step,
                                                                          args['rl_sui'])
                        # print(value_loss, policy_loss)
                        # print()
                        self.rl_update_step += 1
                        if not self.rl_update_step % 20:
                            self.writer.add_scalar('train/value_loss', value_loss.item(), self.rl_update_step)
                            self.writer.add_scalar('train/policy_loss', policy_loss.item(), self.rl_update_step)

            # pdb.set_trace()
            # self.rl_update_step = 0
            if len(result2label) != 0:
                state_pre1 = result2box[:, 8].reshape(-1, 1)
                state_plabel = result2label.reshape(-1, 1)
                state_gtpre1 = torch.tensor([1] * len(gt_labels[1])).cuda().reshape(-1, 1)
                state_gtplabel = gt_labels[1].cuda().reshape(-1, 1)
                # print(state_gtplabel)

                pseudo_masks = [result2gt_masks]

                with torch.no_grad():
                    x = self.model.module.extract_feat(data_batch['img'].data[0].to('cuda'))

                    gt_obbs = gt_mask_bp_obbs_list(gt_masks)  # x,y,w,h,theta
                    gt_rrois = dbbox2roi(
                        [torch.from_numpy(choose_best_Rroi_batch(gt_obbs[1])).to('cuda').type(torch.float32)])
                    gt_rrois[:, 3] = gt_rrois[:, 3] * 1.2
                    gt_rrois[:, 4] = gt_rrois[:, 4] * 1.4
                    gt_rrois[:, 0] = 1
                    gt_rbbox_feats = self.model.module.rbbox_roi_extractor(x[:4],
                                                                           gt_rrois)  # torch.Size([6, 256, 7, 7])
                    if self.model.module.rbbox_head.num_shared_convs > 0:
                        for conv in self.model.module.rbbox_head.shared_convs:
                            gt_rbbox_feats = conv(gt_rbbox_feats)
                    if self.model.module.rbbox_head.num_shared_fcs > 0:
                        if self.model.module.rbbox_head.with_avg_pool:
                            gt_rbbox_feats = self.model.module.rbbox_head.avg_pool(gt_rbbox_feats)
                        gt_rbbox_feats = gt_rbbox_feats.view(gt_rbbox_feats.size(0), -1)
                        for fc in self.model.module.rbbox_head.shared_fcs:
                            gt_rbbox_feats = self.model.module.rbbox_head.relu(
                                fc(gt_rbbox_feats))  # torch.Size([6, 1024])
                    gt_rbbox_feats = F.layer_norm(gt_rbbox_feats, (1024,))
                    gt_rbbox_feats = F.normalize(gt_rbbox_feats, p=2, dim=-1)  # n,1024
                    gt_cls_score = self.model.module.rbbox_head.fc_cls(
                        gt_rbbox_feats) if self.model.module.rbbox_head.with_cls else None
                    gt_score = F.softmax(gt_cls_score * 10, dim=1) if gt_cls_score is not None else None
                    # 二次预测值
                    state_gtpre2 = gt_score.gather(1, gt_labels[1].to('cuda').reshape(-1, 1))

                    pseudo_obbs = gt_mask_bp_obbs_list(pseudo_masks)
                    pseudo_rrois = dbbox2roi(
                        [torch.from_numpy(choose_best_Rroi_batch(pseudo_obbs[0])).to('cuda').type(torch.float32)])
                    pseudo_rrois[:, 3] = pseudo_rrois[:, 3] * 1.2
                    pseudo_rrois[:, 4] = pseudo_rrois[:, 4] * 1.4
                    pseudo_rrois[:, 0] = 1
                    pseudo_rbbox_feats = self.model.module.rbbox_roi_extractor(x[:4],
                                                                               pseudo_rrois)  # torch.Size([6, 256, 7, 7])
                    if self.model.module.rbbox_head.num_shared_convs > 0:
                        for conv in self.model.module.rbbox_head.shared_convs:
                            pseudo_rbbox_feats = conv(pseudo_rbbox_feats)
                    if self.model.module.rbbox_head.num_shared_fcs > 0:
                        if self.model.module.rbbox_head.with_avg_pool:
                            pseudo_rbbox_feats = self.model.module.rbbox_head.avg_pool(pseudo_rbbox_feats)
                        pseudo_rbbox_feats = pseudo_rbbox_feats.view(pseudo_rbbox_feats.size(0), -1)
                        for fc in self.model.module.rbbox_head.shared_fcs:
                            pseudo_rbbox_feats = self.model.module.rbbox_head.relu(
                                fc(pseudo_rbbox_feats))  # torch.Size([6, 1024])
                    pseudo_rbbox_feats = F.layer_norm(pseudo_rbbox_feats, (1024,))
                    pseudo_rbbox_feats = F.normalize(pseudo_rbbox_feats, p=2, dim=-1)  # n,1024
                    pseudo_cls_score = self.model.module.rbbox_head.fc_cls(
                        pseudo_rbbox_feats) if self.model.module.rbbox_head.with_cls else None
                    pseudo_score = F.softmax(pseudo_cls_score * 10, dim=1) if pseudo_cls_score is not None else None
                    # 二次预测值
                    state_pre2 = pseudo_score.gather(1, result2label.to('cuda').reshape(-1, 1))

                    # class_dictionary = torch.load('prototypes.pt')#15,10,1024
                    class_dictionary = self.model.module.prototypes
                    # class_dictionary = F.normalize(class_dictionary, p=2, dim=-1)
                    # 相似度
                    class_dictionary_ave = torch.mean(class_dictionary, dim=1)
                    class_dictionary_ave = class_dictionary_ave / torch.norm(class_dictionary_ave, p=2, dim=1).reshape(
                        (-1, 1))
                    state_sim = torch.sum(
                        torch.mul(pseudo_rbbox_feats, class_dictionary_ave[(result2label - 1).tolist()]),
                        dim=1).reshape(-1, 1)
                    # print(gt_labels[1])
                    # print(gt_rbbox_feats.shape)
                    if gt_labels[1].shape[0] != gt_rbbox_feats.shape[0]:
                        pdb.set_trace()
                        print()
                    state_gtsim = torch.sum(
                        torch.mul(gt_rbbox_feats, class_dictionary_ave[(gt_labels[1].to('cuda') - 1).tolist()]),
                        dim=1).reshape(-1, 1)

                    # 共形
                    sample_cpscore = 1 - state_pre2
                    sample_gtcpscore = 1 - state_gtpre2

                    calibration_cls_score = self.model.module.rbbox_head.fc_cls(
                        class_dictionary.reshape((-1, 1024))) if self.model.module.rbbox_head.with_cls else None
                    calibration_score = F.softmax(calibration_cls_score * 10,
                                                  dim=1) if calibration_cls_score is not None else None
                    calibration_pre = calibration_score.gather(1, torch.tensor(
                        [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10 + [6] * 10 + [7] * 10 + [8] * 10 + [
                            9] * 10 + [10] * 10 + [11] * 10 + [12] * 10 + [13] * 10 + [14] * 10 + [15] * 10).to(
                        'cuda').reshape(-1, 1))
                    calibration_cpscore = 1 - calibration_pre
                    calibration_cpscore = calibration_cpscore.reshape(15, 10)

                    temp = calibration_cpscore[(result2label - 1).tolist()]
                    state_cp = (((temp > sample_cpscore - 0.35) == True).sum(dim=1) / 10).reshape(-1, 1)
                    temp = calibration_cpscore[(gt_labels[1].to('cuda') - 1).tolist()]
                    state_gtcp = (((temp > sample_gtcpscore - 0.35) == True).sum(dim=1) / 10).reshape(-1, 1)

                    state_x = torch.cat((state_pre1, state_pre1, state_sim, state_cp), dim=1)
                    # print(state_gtpre1.shape,state_gtpre2.shape,state_gtsim.shape,state_gtcp.shape,state_gtplabel.shape)
                    state_gt = torch.cat((state_gtpre1, state_gtpre1, state_gtsim, state_gtcp), dim=1)
                    # pdb.set_trace()
                    # print()

                # 基于策略强化网络
                # 记录每个回合的return

                # 存放状态
                index_choose = []
                # 获取初始状态
                state = torch.cat((state_x[0], torch.mean(state_gt[:, 0:4], dim=0)))
                # 开始迭代
                for si in range(len(state_x) - 1):
                    # 动作选择
                    # action = self.agent.take_action(state)  # 对某一状态采取动作
                    action = self.ddpg._actor_forward(state)
                    # action = 1

                    classindex = state_plabel[si].item() - 1
                    choosedgt = torch.cat([calibration_score[classindex * 10:classindex * 10 + 10],
                                           gt_score[(state_gtplabel - 1).reshape(1, -1)[0] == classindex],
                                           pseudo_score[index_choose][
                                               (state_plabel[index_choose] - 1).reshape(1, -1)[0] == classindex]],
                                          dim=0)
                    vector1 = torch.mean(choosedgt, dim=0)
                    vector2 = torch.mean(torch.cat((choosedgt, pseudo_score[si].reshape(-1, 16)), dim=0), dim=0)

                    reward = entropy(vector2) - entropy(vector1) + (vector1[classindex + 1] - vector1[0]) - (
                                vector2[classindex + 1] - vector2[0]) - state_x[si][0].item() / 10 + 0.01
                    reward = reward.float()

                    if action == 1:
                        index_choose.append(si)

                    if action == 1:
                        reward = reward
                    else:
                        reward = -reward

                    if reward < 0:
                        reward = 1.0
                    else:
                        reward = -1.0

                    # 环境更新
                    next_state = torch.cat((state_x[si + 1],
                                            torch.mean(torch.cat((state_gt[:, 0:4], state_x[index_choose][:, :5 - 1])),
                                                       dim=0)), dim=0)

                    print(state_x[si], action, reward)
                    # 更新经验回放池
                    self.replay_buffer.add(state, torch.tensor([action]).cuda(), torch.tensor([reward]).cuda(),
                                           next_state)
                    # training_pair = [state, torch.tensor([action]).cuda(), torch.tensor([reward]).cuda(), next_state]
                    # print(state)
                    state = next_state
                    # 如果经验池超过容量，开始训练
                    if self.replay_buffer.size() > min_size:
                        # 经验池随机采样batch_size组
                        s, a, r, ns = self.replay_buffer.sample(batch_size)
                        # 构造数据集
                        transition_dict = {
                            'states': s,
                            'actions': a,
                            'rewards': r,
                            'next_states': ns,
                        }
                        # pdb.set_trace()
                        # 模型训练
                        value_loss, policy_loss = self.ddpg.update_policy(transition_dict, self.rl_update_step,
                                                                          args['rl_sui'])
                        # print(value_loss, policy_loss)
                        # print(state_x[si],action,reward)
                        # print()
                        self.rl_update_step += 1
                        if not self.rl_update_step % 20:
                            self.writer.add_scalar('train/value_loss', value_loss.item(), self.rl_update_step)
                            self.writer.add_scalar('train/policy_loss', policy_loss.item(), self.rl_update_step)

            # pdb.set_trace()
            if len(data_batch['gt_labels'].data[0][0]) == 0 or len(data_batch['gt_labels'].data[0][1]) == 0:
                continue

            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            # print(self.lut)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')

        self.ddpg.save(
            "/workspace/ReDet/work_dirs/ReDet_re50_refpn_1x_dota1_0.01ddpg/model{}.pkl".format(self._epoch + 1))

        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)
