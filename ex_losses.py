# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.distributed as dist
import numpy as np



class SGLoss(nn.Module):
    def __init__(self, sigma=1, delta=1, view=2, disable_mu=0, topk=10):
        super(SGLoss, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.view = view
        self.disable_mu = disable_mu
        self.topk = topk

    def forward(self, s_emb, t_emb):
        if self.disable_mu:
            s_emb = F.normalize(s_emb)
        t_emb = F.normalize(t_emb)

        N = len(s_emb)
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)

        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb)
            W_P = torch.exp(-T_dist.pow(2) / self.sigma)

            W_P_copy = W_P.clone()
            W_NN = torch.zeros_like(W_P).scatter_(1, torch.topk(W_P_copy, self.topk)[1], torch.ones_like(W_P))
            V = ((W_NN + W_NN.t()) / 2 == 1).float()

            W_C_tilda = torch.zeros_like(W_P)
            for i in range(N):
                indNonzero = torch.where(V[i, :] != 0)[0]
                W_C_tilda[i, indNonzero] = (V[:, indNonzero].sum(1) / len(indNonzero))[indNonzero]

            topk_index = torch.topk(W_P_copy, self.topk)[1]
            topk_half_index = topk_index[:, :int(np.around(self.topk / 2))]
            W_C_hat = W_C_tilda[topk_half_index].mean(1)
            W_C = (W_C_hat + W_C_hat.t()) / 2
            W = (W_P + W_C) / 2

            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)

        pull_losses = torch.relu(S_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - S_dist).pow(2) * neg_weight

        loss = (pull_losses.sum() + push_losses.sum()) / (N * (N - 1))

        return loss

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        # self.ncrops = 1
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)