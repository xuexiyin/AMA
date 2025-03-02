# !/usr/bin/env python
import argparse
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import loader
import builder
from sklearn.metrics.pairwise import cosine_similarity

import vision_transformer as vits
from vision_transformer import DINOHead
from torchvision import models as torchvision_models
import utils
from ex_losses import DINOLoss

from dino_aug import DataAugmentationDINO

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DATASET', default='sketchy', help='sketchy/sketchy2/tuberlin/quickdraw')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_base',
                    choices=['vit_tiny', 'vit_small', 'vit_base'],
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' ')
parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
# parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
#                     metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 2x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=400, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--clean-model', default='./pretrained_dino/dino_vitbase16_pretrain_full_checkpoint.zip', type=str,
                    metavar='PATH',
                    help='path to clean model (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--low-dim', default=768, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')
parser.add_argument('--warmup-epoch', default=20, type=int,
                    help='number of warm-up epochs to only train with self-distillation loss')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use data augmentation')
parser.add_argument('--save-model', action='store_true')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--exp-dir', default='experiment_sketchy', type=str,
                    help='the directory of the experiment')
parser.add_argument('--info-name', default='info', type=str,
                    help='the directory of the experiment')
parser.add_argument('--ckpt-save', default=20, type=int,
                    help='the frequency of saving ckpt')
parser.add_argument('--num-cluster', default='250,500,1000', type=str,
                    help='number of clusters for self entropy loss')


parser.add_argument('--selfentro-temp', default=0.01, type=float,
                    help='the temperature for self-entropy loss')
parser.add_argument('--selfentro-startepoch', default=100, type=int,
                    help='the start epoch for self entropy loss')

parser.add_argument('--sd-weightimage', default=1.0, type=float,
                    help='the weight for self-distillation loss after warm up')
parser.add_argument('--sd-weightsketch', default=2.0, type=float,
                    help='the weight for self-distillation loss after warm up')
parser.add_argument('--cc-weightstart', default=0.0, type=float,
                    help='the starting weight for contrastive clustering loss')
parser.add_argument('--cc-weightsature', default=0.5, type=float,
                    help='the satuate weight for contrastive clustering loss')
parser.add_argument('--cc-startepoch', default=20, type=int,
                    help='the start epoch for contrastive clustering loss')
parser.add_argument('--cc-satureepoch', default=100, type=int,
                    help='the saturated epoch for contrastive clustering loss')

parser.add_argument('--cc-filterthresh', default=0.2, type=float,
                    help='the threshold of filter for contrastive clustering loss')

parser.add_argument('--sg-startepoch', default=40, type=int,
                    help='the start epoch for similarity guidance loss')
parser.add_argument('--sg-weight', default=0.02, type=float,
                    help='the start weight for similarity guidance loss')
parser.add_argument('--ma-startepoch', default=40, type=int,
                    help='the start epoch for mutual alignment loss')
parser.add_argument('--ma-weight', default=0.1, type=float,
                    help='the start weight for mutual alignment loss')

parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
    the DINO head output. For complex and large datasets large values (like 65k) work well.""")
parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                    help="""Whether or not to weight normalize the last layer of the DINO head.
    Not normalizing leads to better performance but can make the training unstable.
    In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
    parameter for teacher update. The value is increased to 1 during training with cosine schedule.
    We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                    help="Whether to use batch normalizations in projection head (Default: False)")

parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                    help="""Initial value for the teacher temperature: 0.04 works well in most cases.
    Try decreasing it if the training loss does not decrease.""")
parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
    of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
    starting with the default value of 0.04 and increase this slightly if needed.""")
parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                    help='Number of warmup epochs for the teacher temperature (Default: 30).')
parser.add_argument("--lr", default=0.0002, type=float, help="""Learning rate at the end of
    linear warmup (highest LR used during training). The learning rate is linearly scaled
    with the batch size, and specified here for a reference batch size of 256.""")
parser.add_argument("--warmup_epochs", default=10, type=int,
                    help="Number of epochs for the linear learning-rate warm up.")
parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
    end of optimization. We use a cosine LR schedule with linear warmup.""")
parser.add_argument('--optimizer', default='adamw', type=str,
                    choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
    weight decay. We use a cosine schedule for WD and using a larger decay by
    the end of training improves performance for ViTs.""")

# Multi-crop parameters
parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.6, 1.),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
    recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
    local views to generate. Set this parameter to 0 to disable multi-crop training.
    When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.2, 0.6),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for small local view cropping of multi-crop.""")
parser.add_argument('--global_crops_scale_sketch', type=float, nargs='+', default=(0.6, 1.),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
    recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")

parser.add_argument('--local_crops_scale_sketch', type=float, nargs='+', default=(0.2, 0.6),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
    Used for small local view cropping of multi-crop.""")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.num_cluster = args.num_cluster.split(',')

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))

    cudnn.benchmark = True
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    transform_sketch = DataAugmentationDINO(
        args.global_crops_scale_sketch,
        args.local_crops_scale_sketch,
        args.local_crops_number,
    )

    train_dataset = loader.TrainDataset(args.data, args.aug_plus, dino_transform=transform,
                                        sketch_transform=transform_sketch)
    eval_dataset = loader.EvalDataset(args.data)
    test_dataset = loader.TestDataset(args.data)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     dino_vitb16 = torch.load('/home/xch/.cache/torch/hub/dino_vitbase16_pretrain_full_checkpoint.zip')

    args.arch = args.arch.replace("deit", "vit")
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    model = builder.AMAModel(
        teacher, student,
        dim=args.low_dim, K_A=eval_dataset.domainA_size, K_B=eval_dataset.domainB_size,
        m=args.moco_m, T=args.temperature, mlp=args.mlp, selfentro_temp=args.selfentro_temp,
        num_cluster=args.num_cluster, cc_filterthresh=args.cc_filterthresh)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.clean_model:
        if os.path.isfile(args.clean_model):
            print("=> loading pretrained clean model '{}'".format(args.clean_model))

            loc = 'cuda:{}'.format(args.gpu)
            clean_checkpoint = torch.load(args.clean_model, map_location=loc)

            current_state = model.state_dict()
            used_pretrained_state = {}

            for k in current_state:
                if 'student' in k:
                    k_parts = '.'.join(k.split('.')[1:])
                    if k_parts.endswith("_g"):
                        continue
                    if k_parts.endswith("_v"):
                        k_parts = k_parts[:-2]
                    used_pretrained_state[k] = clean_checkpoint['student']['module.' + k_parts]
                elif 'teacher' in k:
                    k_parts = '.'.join(k.split('.')[1:])
                    if k_parts.endswith("_g"):
                        continue
                    if k_parts.endswith("_v"):
                        k_parts = k_parts[:-2]
                    used_pretrained_state[k] = clean_checkpoint['teacher'][k_parts]
                else:
                    print(k)
            current_state.update(used_pretrained_state)
            model.load_state_dict(current_state)
        else:
            print("=> no clean model found at '{}'".format(args.clean_model))

    info_save = open(os.path.join(args.exp_dir, 'info_{}.txt'.format(args.info_name)), 'w')
    best_eval = [0., 0., 0., 0., 0]
    for epoch in range(args.epochs):

        cluster_result = None
        if epoch >= args.warmup_epoch:
            features_A, features_B, _, _ = compute_features(eval_loader, model, args)
            features_A = features_A.numpy()
            features_B = features_B.numpy()
            cluster_result = run_kmeans(features_A, features_B, args)


        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args, info_save, cluster_result, dino_loss=dino_loss)
        if args.save_model:
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion}, '{}/{}checkpoint{}.pth.tar'.format(args.exp_dir, args.info_name, epoch))

        features_A, features_B, targets_A, targets_B = compute_features(test_loader, model, args)
        features_A = features_A.numpy()
        targets_A = targets_A.numpy()

        mapall, map200, prec100, prec200 = retrieval_precision_cal(features_A, targets_A, features_B, targets_B)
        print("mAP@all: {};   mAP@200: {};   Prec@100: {} ;   Prec@200: {} \n".format(mapall, map200, prec100, prec200))
        info_save.write(
            "mAP@all: {};   mAP@200: {};   Prec@100: {} ;   Prec@200: {} \n".format(mapall, map200, prec100, prec200))

        if best_eval[0] < mapall:
            best_eval[0], best_eval[1], best_eval[2], best_eval[3], best_eval[
                4] = mapall, map200, prec100, prec200, epoch

    info_save.write(
        "mAP@all: {};   mAP@200: {};   Prec@100: {} ;   Prec@200: {} ;   bestepoch: {} \n".format(best_eval[0],
                                                                                                  best_eval[1],
                                                                                                  best_eval[2],
                                                                                                  best_eval[3],
                                                                                                  best_eval[4]))


def train(train_loader, model, criterion, optimizer, epoch, args, info_save, cluster_result, dino_loss=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = {'SD_A': AverageMeter('Loss_Self_Distil_A', ':.4e'),
              'SD_B': AverageMeter('Loss_Self_Distil_B', ':.4e'),
              'Total_loss': AverageMeter('Loss_Total', ':.4e')}

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time,
         losses['Total_loss'],
         losses['SD_A'], losses['SD_B']],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_A, image_ids_A, images_B, image_ids_B, cates_A, cates_B) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.gpu is not None:
            images_A = [im.cuda(non_blocking=True) for im in images_A]
            image_ids_A = image_ids_A.cuda(args.gpu, non_blocking=True)

            images_B = [im.cuda(non_blocking=True) for im in images_B]
            image_ids_B = image_ids_B.cuda(args.gpu, non_blocking=True)

        if epoch >= args.ma_startepoch:
            stage = 'cross_alignment'
        else:
            stage = 'represent_learning'

        losses_sd, \
            q_A, q_B, \
            losses_ma, \
            losses_cc,losses_sg  = model(im_q_A=images_A,
                                                im_id_A=image_ids_A, im_q_B=images_B,
                                                im_id_B=image_ids_B,
                                                cluster_result=cluster_result,
                                                criterion=criterion, stage=stage, dino_loss=dino_loss, epoch=epoch)

        sd_loss_A = losses_sd['domain_A']
        sd_loss_B = losses_sd['domain_B']

        losses['SD_A'].update(sd_loss_A.item(), images_A[0].size(0))
        losses['SD_B'].update(sd_loss_B.item(), images_B[0].size(0))

        loss_A = sd_loss_A * args.sd_weightsketch
        loss_B = sd_loss_B * args.sd_weightimage

        if epoch >= args.cc_startepoch:

            cc_loss_A = losses_cc['domain_A']
            cc_loss_B = losses_cc['domain_B']

            losses['CC_A'].update(cc_loss_A.item(), images_A[0].size(0))
            losses['CC_B'].update(cc_loss_B.item(), images_B[0].size(0))

            if epoch <= args.cc_startepoch:
                cur_cc_weight = args.cc_weightstart
            elif epoch < args.sg_startepoch:
                cur_cc_weight = args.ccweightstart + (args.cc_weightsature - args.cc_weightstart) * \
                                   ((epoch - args.cc_startepoch) / (args.sg_satureepoch - args.cc_startepoch))
            else:
                cur_cc_weight = args.cc_weightsature

            loss_A += cc_loss_A * cur_cc_weight
            loss_B += cc_loss_B * cur_cc_weight

        all_loss = loss_A + loss_B

        if epoch >= args.sg_startepoch:

            all_loss += losses_sg['domain_A'] * args.sg_weight
            all_loss += losses_sg['domain_B'] * args.sg_weight
        if epoch >= args.ma_startepoch:

            losses_distlogits_list = []
            for key in losses_distlogits.keys():
                losses_distlogits_list.extend(losses_distlogits[key])

            losses_distlogits_mean = torch.mean(torch.stack(losses_distlogits_list))
            losses['MA'].update(losses_distlogits_mean.item(), images_A[0].size(0))

            all_loss += losses_distlogits_mean * args.ma_weight

        losses['Total_loss'].update(all_loss.item(), images_A[0].size(0))

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            info = progress.display(i)
            info_save.write(info + '\n')


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()

    features_A = torch.zeros(eval_loader.dataset.domainA_size, args.low_dim).cuda()
    features_B = torch.zeros(eval_loader.dataset.domainB_size, args.low_dim).cuda()

    targets_all_A = torch.zeros(eval_loader.dataset.domainA_size, dtype=torch.int64).cuda()
    targets_all_B = torch.zeros(eval_loader.dataset.domainB_size, dtype=torch.int64).cuda()

    for i, (images_A, indices_A, targets_A, images_B, indices_B, targets_B) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images_A = images_A.cuda(non_blocking=True)
            images_B = images_B.cuda(non_blocking=True)
            indices_A = indices_A.type(torch.long)
            indices_B = indices_B.type(torch.long)
            targets_A = targets_A.cuda(non_blocking=True)
            targets_B = targets_B.cuda(non_blocking=True)

            feats_A, feats_B = model(im_q_A=images_A, im_q_B=images_B, is_eval=True)

            features_A[indices_A] = feats_A
            features_B[indices_B] = feats_B

            targets_all_A[indices_A] = targets_A
            targets_all_B[indices_B] = targets_B

    return features_A.cpu(), features_B.cpu(), targets_all_A.cpu(), targets_all_B.cpu()


def run_kmeans(x_A, x_B, args):
    print('performing kmeans clustering')
    results = {'im2cluster_A': [], 'centroids_A': [],
               'im2cluster_B': [], 'centroids_B': []}
    for domain_id in ['A', 'B']:
        if domain_id == 'A':
            x = x_A
        elif domain_id == 'B':
            x = x_B
        else:
            x = np.concatenate([x_A, x_B], axis=0)

        for seed, num_cluster in enumerate(args.num_cluster):
            # intialize faiss clustering parameters
            d = x.shape[1]
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed
            clus.max_points_per_centroid = 2000
            clus.min_points_per_centroid = 2
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = args.gpu
            index = faiss.IndexFlatL2(d)

            clus.train(x, index)
            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).cuda()
            centroids_normed = nn.functional.normalize(centroids, p=2, dim=1)
            im2cluster = torch.LongTensor(im2cluster).cuda()

            results['centroids_' + domain_id].append(centroids_normed)
            results['im2cluster_' + domain_id].append(im2cluster)

    return results



def retrieval_precision_cal(predicted_features_query, gt_labels_query, predicted_features_gallery, gt_labels_gallery):
    scores = cosine_similarity(predicted_features_query, predicted_features_gallery)

    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mAP_ls2 = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery)
        mapi2 = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, 200)
        mAP_ls[gt_labels_query[fi]].append(mapi)
        mAP_ls2[gt_labels_query[fi]].append(mapi2)

    prec_ls1 = [[] for _ in range(len(np.unique(gt_labels_query)))]
    prec_ls2 = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        prec1 = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery, 100)
        prec2 = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery, 200)
        prec_ls1[gt_labels_query[fi]].append(prec1)
        prec_ls2[gt_labels_query[fi]].append(prec2)
    return np.nanmean(sum(mAP_ls, [])), np.nanmean(sum(mAP_ls2, [])), np.nanmean(sum(prec_ls1, [])), np.nanmean(
        sum(prec_ls2, []))


def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap


def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.5 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
