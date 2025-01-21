# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 16:29
# @Author : 李昌杏
# @File : train_teacher.py
# @Software : PyCharm

import argparse
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_data
from network.teacher import ModalityFusionNetwork
from utils.losses import MOCO
import os
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
        parser.add_argument("--num_epochs", default=40)
        parser.add_argument("--batch_size", default=48)
        parser.add_argument("--lr", default=3e-6)
        parser.add_argument("--weight_decay", default= 1e-4)
        #net
        parser.add_argument("--teacher_out_dim", default=768)
        parser.add_argument("--teacher_encoder", default='vit_base_patch16_224')
        parser.add_argument("--VIT_pre_weight", default='vit.npz')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

def main(args):
    writer = SummaryWriter(f'log_{args.dataset}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[INFO] Setting SEED: " + str(args.seed))

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    teacher_out_dim=args.teacher_out_dim

    datasets, sk_valid_data, im_valid_data = load_data(args)
    num_class=datasets.get_num_class()
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,drop_last=True)

    teacher = ModalityFusionNetwork(768, 3,  heads=3,num_class=num_class,
                                    encoder_backbone=args.teacher_encoder,checkpoint_path=args.VIT_pre_weight).cuda()
    lr = args.lr
    weight_decay = args.weight_decay
    fp16_precision = True
    optimizer = torch.optim.Adam(teacher.parameters(), lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=fp16_precision)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs, eta_min=0,
                                                           last_epoch=-1)
    loss_moco_sk=MOCO(K=int(batch_size*256),dim=teacher_out_dim).cuda()
    loss_moco_im=MOCO(K=int(batch_size*256),dim=teacher_out_dim).cuda()
    ce_criterion = torch.nn.CrossEntropyLoss().cuda()

    min=999999
    index=0
    for epoch in range(num_epochs):

        epoch_train_contrastive_loss = 0

        for batch_idx, data in enumerate(tqdm(train_loader)):
            index+=1
            teacher.train()
            sk, im, sketch_neg, image_neg,label,label_neg= data
            sk, im,label = sk.cuda(), im.cuda(),label.cuda()

            optimizer.zero_grad()

            photo1_cls,sketch1_cls,photo1_f,sketch1_f = teacher(im, sk)

            teacher_loss = ce_criterion(*loss_moco_im(photo1_cls,sketch1_cls))+\
                           ce_criterion(*loss_moco_sk(sketch1_cls,photo1_cls))
            class_loss=ce_criterion(photo1_f,label)+ce_criterion(sketch1_f,label)

            loss = teacher_loss/2+class_loss/2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_contrastive_loss = epoch_train_contrastive_loss + teacher_loss.item()+class_loss.item()

            teacher.eval()
            if batch_idx % 10 == 0:
                scheduler.step()

            writer.add_scalar("train/moco_loss", teacher_loss, index)
            writer.add_scalar("train/class_loss", class_loss, index)

        writer.add_scalar("train/loss", epoch_train_contrastive_loss, epoch + 1)
        writer.add_scalar("train/avg_loss", epoch_train_contrastive_loss/len(train_loader), epoch + 1)

        print('Epoch Train: [', epoch, '] Contrast Loss: ', epoch_train_contrastive_loss,'avg Contrast Loss: ',epoch_train_contrastive_loss/len(train_loader))

        if epoch_train_contrastive_loss<min:
            min=epoch_train_contrastive_loss
            torch.save(teacher.state_dict(),f'weights/teacher_{args.dataset}_{args.teacher_encoder}_good.pth')
        #
        # if (epoch+1)%1 == 0:
        #     torch.save(teacher.state_dict(),f'weights/teacher_{args.dataset}_{args.teacher_encoder}_{epoch+1}.pth')


if __name__ == '__main__':
    main(Option().parse())