import copy

import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from ex_losses import SGLoss


class AMAModel(nn.Module):

    def __init__(self, teacher=None, student=None, dim=128, K_A=65536, K_B=65536,
                 m=0.999, T=0.1, mlp=False, selfentro_temp=0.2,
                 num_cluster=None, cc_filterthresh=0.2):

        super(AMAModel, self).__init__()

        self.K_A = K_A
        self.K_B = K_B
        self.m = m
        self.T = T

        self.selfentro_temp = selfentro_temp
        self.num_cluster = num_cluster
        self.cc_filterthresh = cc_filterthresh

        norm_layer = partial(SplitBatchNorm, num_splits=2)

        self.student = student
        self.teacher = teacher

        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, im_q_A, im_q_B, im_id_A=None,
                im_id_B=None, is_eval=False,
                cluster_result=None, criterion=None, stage='contra', dino_loss=None, epoch=None):
        if is_eval:

            k_A = self.teacher(im_q_A, eval=True)
            k_B = self.teacher(im_q_B, eval=True)
            return k_A, k_B
        q_A = self.student(im_q_A, eval=True)
        q_B = self.student(im_q_B, eval=True)
        k_A = self.teacher(im_q_A[:2], eval=True)
        k_B = self.teacher(im_q_B[:2], eval=True)
        with torch.no_grad():
            self._momentum_update_key_encoder()

        loss_dino_A = dino_loss(self.upsample_prostu(q_A), self.upsample_protea(k_A), epoch)
        loss_dino_B = dino_loss(self.upsample_prostu(q_B), self.upsample_protea(k_B), epoch)
        losses_dino = {'domain_A': loss_dino_A,
                       'domain_B': loss_dino_B}


        if cluster_result is not None and stage == 'cross_alignment':
            q_A = F.normalize(q_A, dim=1)
            q_B = F.normalize(q_B, dim=1)
            k_A = F.normalize(k_A, dim=1)
            k_B = F.normalize(k_B, dim=1)

            loss_cc_A, \
                loss_cc_B = self.cluster_contrastive_loss_new(q_A, k_A, im_id_A,
                                                                 q_B, k_B, im_id_B,
                                                                 cluster_result)

            losses_cc = {'domain_A': loss_cc_A,
                            'domain_B': loss_cc_B}

            losses_ma = self.dist_of_logit_loss(q_A, q_B, cluster_result, self.num_cluster)
            sg_loss = SGLoss()
            sg_lossA = sg_loss(q_A, k_A)
            sg_lossB = sg_lossA(k_A, k_B)
            losses_sg = {'domain_A': sg_lossA,
                          'domain_B': sg_lossB}

            return losses_dino, q_A, q_B, losses_ma, losses_cc, losses_sg
        elif cluster_result is not None and stage == 'represent_learning':
            q_A = F.normalize(q_A, dim=1)
            q_B = F.normalize(q_B, dim=1)
            k_A = F.normalize(k_A, dim=1)
            k_B = F.normalize(k_B, dim=1)
            loss_cc_A, \
                loss_cc_B = self.cluster_contrastive_loss(q_A, k_A, im_id_A,
                                                                 q_B, k_B, im_id_B,
                                                                 cluster_result)

            losses_cc = {'domain_A': loss_cc_A,
                            'domain_B': loss_cc_B}
            return losses_dino, q_A, q_B, None, None, losses_cc, None
        else:
            return losses_dino, None, None, None, None, None

    def cluster_contrastive_loss(self, q_A, k_A, im_id_A, q_B, k_B, im_id_B, cluster_result):

        all_losses = {'domain_A': [], 'domain_B': []}
        batch_size = k_B.shape[0] // 2
        n_crop = q_A.shape[0] // batch_size - 2

        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                im_id = im_id_A
                k_feat = k_A
                k_feat = k_feat.chunk(2)
                q_feat = q_A


            else:
                im_id = im_id_B
                k_feat = k_B
                k_feat = k_feat.chunk(2)
                q_feat = q_B
            mask = 1.0
            for n, (im2cluster, prototypes) in enumerate(zip(cluster_result['im2cluster_' + domain_id],
                                                             cluster_result['centroids_' + domain_id])):
                cor_cluster_id = im2cluster[im_id]

                mask *= torch.eq(cor_cluster_id.contiguous().view(-1, 1),
                                 cor_cluster_id.contiguous().view(1, -1)).float()  # batch size x batch size
                base_mask = copy.deepcopy(mask)
                for _ in range(n_crop):
                    mask = torch.cat([mask, base_mask], dim=1)

                all_score = torch.div(torch.matmul(q_feat[:batch_size], q_feat[batch_size:].T), self.T)

                exp_all_score = torch.exp(all_score)

                log_prob = all_score - torch.log(exp_all_score.sum(1, keepdim=True))

                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

                cor_proto = prototypes[cor_cluster_id]
                inst_pos_value = torch.exp(
                    torch.div(torch.einsum('nc,nc->n', [k_feat[0], cor_proto]), self.T))  # N
                inst_all_value = torch.exp(
                    torch.div(torch.einsum('nc,ck->nk', [k_feat[0], prototypes.T]), self.T))  # N x r
                filters = ((inst_pos_value / torch.sum(inst_all_value, dim=1)) > self.cc_filterthresh).float()
                filters_sum = filters.sum()

                loss = - (filters * mean_log_prob_pos).sum() / (filters_sum + 1e-8)

                all_losses['domain_' + domain_id].append(loss)

        return torch.mean(torch.stack(all_losses['domain_A'])), torch.mean(torch.stack(all_losses['domain_B']))


    def dist_of_logit_loss(self, q_A, q_B, cluster_result, num_cluster):
        q_A = q_A[0:q_A.shape[0]/5]
        q_B = q_B[0:q_B.shape[0]/5]
        all_losses = {}

        for n, (proto_A, proto_B) in enumerate(zip(cluster_result['centroids_A'],
                                                   cluster_result['centroids_B'])):

            if str(proto_A.shape[0]) in num_cluster:
                domain_ids = ['A', 'B']

                for domain_id in domain_ids:
                    if domain_id == 'A':
                        feat = q_A
                    elif domain_id == 'B':
                        feat = q_B
                    else:
                        feat = torch.cat([q_A, q_B], dim=0)

                    loss_A_B = self.dist_of_dist_loss_onepair(feat, proto_A, proto_B)

                    key_A_B = 'feat_domain_' + domain_id + '_A_B' + '-cluster_' + str(proto_A.shape[0])
                    if key_A_B in all_losses.keys():
                        all_losses[key_A_B].append(loss_A_B.mean())
                    else:
                        all_losses[key_A_B] = [loss_A_B.mean()]

        return all_losses

    def dist_of_dist_loss_onepair(self, feat, proto_1, proto_2):

        proto1_distlogits = self.dist_cal(feat, proto_1)
        proto2_distlogits = self.dist_cal(feat, proto_2)

        loss_A_B = F.pairwise_distance(proto1_distlogits, proto2_distlogits, p=2) ** 2

        return loss_A_B

    def dist_cal(self, feat, proto, temp=0.01):

        proto_logits = F.softmax(torch.matmul(feat, proto.T) / temp, dim=1)

        proto_distlogits = 1.0 - torch.matmul(F.normalize(proto_logits, dim=1), F.normalize(proto_logits.T, dim=0))

        return proto_distlogits

    def upsample_protea(self, pro):
        return self.teacher.head(pro)

    def upsample_prostu(self, pro):
        return self.student.head(pro)


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape

        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)

            outcome = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)
