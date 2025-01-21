# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 16:40
# @Author : 李昌杏
# @File : train_student.py
# @Software : PyCharm
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import utils.utils as utils
import network.student as vits
from network.teacher import ModalityFusionNetwork
from datasets import load_data
import utils.losses as losses
from test import valid
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")
        # dataset
        parser.add_argument('--data_path', type=str, default="../root/autodl-tmp")
        parser.add_argument('--dataset_len', type=int, default=10000)#choices=[149428//2,55252]
        parser.add_argument('--dataset', type=str, default='tu_berlin',
                            choices=['Sketchy21', 'tu_berlin', 'Quickdraw','Sketchy25'])
        parser.add_argument('--test_class', type=str, default='test_class_tuberlin30',
                            choices=['test_class_sketchy25', 'test_class_sketchy21', 'test_class_tuberlin30', 'Quickdraw'])
        parser.add_argument('--testall', default=False, action='store_true', help='train/test scale')
        parser.add_argument("--seed", default=0)
        #train
        parser.add_argument("--img_size", default=224)
        parser.add_argument("--epoch", default=50)
        parser.add_argument("--warmup_epochs", default=5)
        parser.add_argument("--batch_size", default=32)
        parser.add_argument("--stu_batch_size", default=128)
        parser.add_argument("--lr", default=3e-6)
        parser.add_argument("--min_lr", default=1e-6)
        parser.add_argument("--weight_decay", default= 0.04)
        parser.add_argument("--weight_decay_end", default=0.4)
        parser.add_argument("--MOCO_K", default= 16384)
        #net
        parser.add_argument("--teacher_out_dim", default=768)
        parser.add_argument("--student_dim", default=768)
        parser.add_argument("--teacher_encoder", default='vit_base_patch16_224')
        parser.add_argument("--VIT_pre_weight", default='vit.npz')
        parser.add_argument("--teacher_pre_weight", default='vit.npz')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
def main():
    args=Option().parse()
    writer = SummaryWriter(f'log_{args.dataset}_student')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[INFO] Setting SEED: " + str(args.seed))
    #######################data initial
    datasets, sk_valid_data, im_valid_data = load_data(args=args)
    num_class=datasets.get_num_class()
    train_loader = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,drop_last=True)
    skt_loader = DataLoader(sk_valid_data, batch_size=args.stu_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    img_loader = DataLoader(im_valid_data, batch_size=args.stu_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    #######################model initial
    student_img = vits.Student_IMG(datasets.get_num_class(),checkpoint_path=args.VIT_pre_weight,feature_dim=args.student_dim).cuda()
    student_skt = vits.Student_SKT(datasets.get_num_class(),checkpoint_path=args.VIT_pre_weight,feature_dim=args.student_dim).cuda()
    teacher = ModalityFusionNetwork(768, 3, heads=3,encoder_backbone=args.teacher_encoder,
                                    num_class=num_class,
                                    checkpoint_path=args.teacher_pre_weight).cuda()
    teacher.load_state_dict(torch.load(f'weights/teacher_{args.dataset}_{args.teacher_encoder}_good.pth'))
    # there is no backpropagation through the teacher, so no need for gradients
    for _ in teacher.parameters():
        _.requires_grad = False
    #######################loss initial
    loss_cn = torch.nn.CrossEntropyLoss().cuda()
    ca_loss = losses.CenterAlignment(datasets.get_num_class(),fea_dim=256).cuda()
    loss_moco_sk= losses.MOCO(K=args.MOCO_K,dim=num_class).cuda()
    loss_moco_im= losses.MOCO(K=args.MOCO_K,dim=num_class).cuda()
    #######################optimizer initial
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer_img = torch.optim.AdamW(student_img.parameters(), lr, weight_decay=weight_decay)
    optimizer_skt = torch.optim.AdamW(student_skt.parameters(), lr, weight_decay=weight_decay)
    #######################scheduler initial
    loss_scheduler = utils.cosine_scheduler_loss(0, 2, args.epoch, len(train_loader),
                                                 warmup_epochs=5, early_schedule_epochs=30)
    lr_schedule = utils.cosine_scheduler(
        lr * (args.batch_size) / 256.,  # linear scaling rule
        args.min_lr,
        args.epoch, len(train_loader),
        warmup_epochs=args.warmup_epochs,
        early_schedule_epochs=0,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,  # linear scaling rule
        args.weight_decay_end,
        args.epoch, len(train_loader),
        warmup_epochs=args.warmup_epochs,
        early_schedule_epochs=0,
    )
    fp16_scaler = torch.cuda.amp.GradScaler()

    min_loss = -1
    index = 0

    for epoch in range(args.epoch):
        epoch_train_contrastive_loss = 0
        for batch_idx, data in enumerate(tqdm(train_loader)):

            it = len(train_loader)* epoch + batch_idx # global training iteration
            for i, param_group in enumerate(optimizer_skt.param_groups):
                if i == 0 or i == 1:
                    param_group['lr'] = lr_schedule[it] * 0.1
                else:
                    param_group["lr"] = lr_schedule[it]
                if i == 0 or i == 2:  # only the first group is regularized; look at get_params_groups for details
                    param_group["weight_decay"] = wd_schedule[it]
            for i, param_group in enumerate(optimizer_img.param_groups):
                if i == 0 or i == 1:
                    param_group['lr'] = lr_schedule[it] * 0.1
                else:
                    param_group["lr"] = lr_schedule[it]
                if i == 0 or i == 2:  # only the first group is regularized; look at get_params_groups for details
                    param_group["weight_decay"] = wd_schedule[it]

            sketch, photo, sketch_neg, photo_neg,label,label_neg=data
            sketch, photo, sketch_neg, photo_neg, label, label_neg=\
                sketch.cuda(), photo.cuda(), sketch_neg.cuda(), photo_neg.cuda(), label.cuda(), label_neg.cuda()

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                optimizer_img.zero_grad()
                optimizer_skt.zero_grad()

                photo1_cls, sketch1_cls, photo1_fea, sketch1_fea = teacher(photo, sketch)
                photo1_cls, sketch1_cls, photo1_fea, sketch1_fea = \
                    photo1_cls.detach(), sketch1_cls.detach(), photo1_fea.detach(), sketch1_fea.detach()
                photo1_neg_cls, sketch1_neg_cls, photo1_neg_f, sketch1_neg_f = teacher(photo_neg, sketch_neg)
                photo1_neg_cls, sketch1_neg_cls, photo1_neg_f, sketch1_neg_f = \
                    photo1_neg_cls.detach(), sketch1_neg_cls.detach(), photo1_neg_f.detach(), sketch1_neg_f.detach()

                skt_logits,skt_fea,skt_token = student_skt(sketch)# class_logits,representation,cls_token,
                skt_neg_logits,skt_neg_fea,skt_neg_token = student_skt(sketch_neg)

                img_logits,img_fea,img_token= student_img(photo)
                img_neg_logits,img_neg_fea,img_neg_token= student_img(photo_neg)

                center_align_loss = (ca_loss(skt_fea, label, 'skt')+ca_loss(skt_neg_fea, label_neg, 'skt')+
                                     ca_loss(img_fea, label, 'img') + ca_loss(img_neg_fea, label_neg, 'img')
                                     ) * loss_scheduler[it]/4

                q_loss = losses.q_loss(skt_token, skt_neg_token, img_token,img_neg_token)

                cn_loss = (loss_cn(skt_logits, label) + loss_cn(skt_neg_logits, label_neg) +
                           loss_cn(img_logits, label) +
                           loss_cn(img_neg_logits, label_neg)) / 4

                kld_loss=(loss_cn(*losses.sim_moco(skt_logits,sketch1_fea,loss_moco_sk))+
                          loss_cn(*losses.sim_moco(skt_neg_logits,sketch1_neg_f,loss_moco_sk))+
                          loss_cn(*losses.sim_moco(img_logits,photo1_fea,loss_moco_im))+
                          loss_cn(*losses.sim_moco(img_neg_logits,photo1_neg_f,loss_moco_im)))/4

                loss = q_loss + cn_loss + kld_loss+center_align_loss

                epoch_train_contrastive_loss += loss.item()

                fp16_scaler.scale(loss).backward()
                index += 1
                writer.add_scalar("train_student/q", q_loss, index)
                writer.add_scalar("train_student/cn", cn_loss, index)
                writer.add_scalar("train_student/kld", kld_loss, index)
                writer.add_scalar("train_student/center",center_align_loss , index)

                fp16_scaler.step(optimizer_skt)
                fp16_scaler.step(optimizer_img)
                fp16_scaler.update()

        writer.add_scalar("train_student/loss", epoch_train_contrastive_loss, epoch + 1)
        writer.add_scalar("train_student/avg_loss", epoch_train_contrastive_loss , epoch + 1)
        if epoch>=4:
            acc=valid(skt_loader,img_loader,student_img,student_skt)
            writer.add_scalar("train_student/acc",acc , epoch + 1)
            print('Epoch Train: [', epoch, '] acc: ',acc,' Loss: ', epoch_train_contrastive_loss, 'avg Loss: ',
                  epoch_train_contrastive_loss / len(train_loader))
            if  acc > min_loss:
                min_loss = acc
                torch.save(student_img.state_dict(), f'weights/student_{args.dataset}_img_good.pth')
                torch.save(student_skt.state_dict(), f'weights/student_{args.dataset}_skt_good.pth')
        else:
            print('Epoch Train: [', epoch, '] Loss: ', epoch_train_contrastive_loss, 'avg Loss: ',
                  epoch_train_contrastive_loss / len(train_loader))

if __name__ == '__main__':
    main()