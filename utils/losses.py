# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 16:30
# @Author : 李昌杏
# @File : losses.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
def q_loss(sk_1,sk_2,im_1,im_2):
    triplet = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
    loss = triplet(sk_1, im_1, im_2)
    loss += triplet(sk_1, im_1, sk_2)

    return loss/2

def kld(target,input,tmp):
    target=target/tmp
    input=input/tmp
    target=F.softmax(target,dim=1)
    input=F.log_softmax(input,dim=1)
    l_kl=F.kl_div(input,target,size_average=False)/input.shape[0]
    return l_kl

class MOCO(nn.Module):
    """
    Based on the MoCo Loss.
    GitHub: https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    Paper: https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=768, K=1024, m=0.999, T=0.07):
        super(MOCO, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        if self.K % batch_size != 0:
            return

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, query, key):
        with torch.no_grad():  # no gradient to keys
            key = nn.functional.normalize(key, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [query, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(key)

        return logits, labels

def sim_moco(query, key,moco:MOCO,tmp=0.05):
    label = torch.argmax(key,dim=1)
    return query,label
def sim_moco1(query, key,moco:MOCO,tmp=0.05):
    query = query
    key = key
    query = F.softmax(query, dim=1)
    key = F.softmax(key, dim=1)
    return moco(query,key)
def align_loss(x, y, alpha=2):
    '''
    https://github.com/SsnL/align_uniform/blob/master/align_uniform/__init__.py
    :param x:
    :param y:
    :param alpha:
    :return:
    '''
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    '''
    https://github.com/SsnL/align_uniform/blob/master/align_uniform/__init__.py
    :param x:
    :param t:
    :return:
    '''
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class CenterAlignment(nn.Module):
    def __init__(self,num_classes=104, fea_dim=256, momentum=0.9):
        super(CenterAlignment, self).__init__()
        self.num_classes = num_classes
        self.fea_dim = fea_dim
        self.momentum = momentum
        self.register_buffer("center_skt", torch.zeros(num_classes, fea_dim))
        self.register_buffer("center_img", torch.zeros(num_classes, fea_dim))

    def forward(self, x, l, modality='img'):
        class_in_batch = self.update_center(x, l, modality)

        return align_loss(self.center_img[class_in_batch], self.center_skt[class_in_batch])

    def update_center(self, x, l, modality):
        self.center_img = self.center_img.detach()
        self.center_skt = self.center_skt.detach()

        all_l=l
        classes_in_batch, sam2cls_idx, cl_sam_counts = torch.unique(all_l, return_counts=True, sorted=True, return_inverse=True)
        center_tmp = torch.zeros(len(classes_in_batch), self.fea_dim).cuda()
        for i, idx in enumerate(sam2cls_idx):
            center_tmp[idx] += x[i]
        center_tmp = center_tmp / cl_sam_counts.unsqueeze(1)

        if modality == 'img':
            self.center_img[classes_in_batch] = self.center_img[classes_in_batch] * self.momentum + center_tmp * (1 - self.momentum)
            self.center_img[classes_in_batch] /= self.center_img[classes_in_batch].norm(p=2, dim=1, keepdim=True)
        else:
            self.center_skt[classes_in_batch] = self.center_skt[classes_in_batch] * self.momentum + center_tmp * (1 - self.momentum)
            self.center_skt[classes_in_batch] /= self.center_skt[classes_in_batch].norm(p=2, dim=1, keepdim=True)

        return classes_in_batch