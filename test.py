# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 22:58
# @Author : 李昌杏
# @File : test.py
# @Software : PyCharm
import argparse
import multiprocessing
import torch
import torch.utils.data
from joblib import Parallel, delayed
from torch import nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import network.student as vits
from datasets import load_data
import numpy as np
import torch.nn.functional as F
from utils.utils import map_sake, prec_sake

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def compute_avg_prec(sketch_label, retrieval):
    num_correct = 0
    avg_prec = 0
    for photo_idx, photo_class in enumerate(retrieval, start=1):
        if photo_class == sketch_label:
            num_correct += 1
            avg_prec = avg_prec + (num_correct / photo_idx)

    if num_correct > 0:
        avg_prec = avg_prec / num_correct

    return avg_prec

def mAPcuda(photo_loader, sketch_loader, model_img,model_skt ,k=None):
    num_cores = min(multiprocessing.cpu_count(), 32)
    gallery_reprs = []
    gallery_labels = []
    model_skt.eval()
    model_img.eval()
    with torch.no_grad():
        for photo, label in photo_loader:
            photo, label = photo.cuda(), label.cuda()
            photo_reprs = model_img.embedding(photo)
            gallery_reprs.append(photo_reprs)
            gallery_labels.append(label)

        gallery_reprs = F.normalize(torch.cat(gallery_reprs))
        gallery_labels = torch.cat(gallery_labels)

        aps_all = []
        for sketch, label in sketch_loader:
            sketch, label = sketch.cuda(), label
            sketch_reprs = F.normalize(model_skt.embedding(sketch))
            # sketch_reprs = model.embedding(sketch,'skt')[2]
            ranks = torch.argsort(torch.matmul(sketch_reprs, gallery_reprs.T), dim=1, descending=True)
            # num_correct = torch.sum(gallery_labels[ranks[:, 0]] == label).item()
            retrievals = gallery_labels[ranks]
            if k is not None:
                retrievals = gallery_labels[ranks[:, :k]]

            aps = Parallel(n_jobs=num_cores)(
                delayed(compute_avg_prec)(label[sketch_idx].item(), retrieval.cpu().numpy()) for sketch_idx, retrieval
                in enumerate(retrievals))
            aps_all.extend(aps)

        return np.mean(aps_all)

def mAP1(photo_loader, sketch_loader, model_img,model_skt):
    gallery_reprs = []
    gallery_reprs_skt = []
    gallery_labels = []
    gallery_labels_skt = []
    model_skt.eval()
    model_img.eval()
    with torch.no_grad():
        for idx,(photo, label) in enumerate(tqdm(photo_loader)):
            photo, label = photo.cuda(), label
            photo_reprs = model_img.embedding(photo).cpu()
            gallery_reprs.append(photo_reprs)
            gallery_labels.append(label)

        gallery_reprs = F.normalize(torch.cat(gallery_reprs))
        gallery_labels = torch.cat(gallery_labels)

        for idx,(sketch, label) in enumerate(tqdm(sketch_loader)):
            sketch, label = sketch.cuda(), label
            sketch_reprs = F.normalize(model_skt.embedding(sketch)).cpu()
            gallery_reprs_skt.append(sketch_reprs)
            gallery_labels_skt.append(label)

        gallery_reprs_skt = F.normalize(torch.cat(gallery_reprs_skt))
        gallery_labels_skt = torch.cat(gallery_labels_skt)

    test_features_img = nn.functional.normalize(gallery_reprs, dim=1, p=2)
    test_features_skt = nn.functional.normalize(gallery_reprs_skt, dim=1, p=2)
    ############################################################################
    # Step 2: similarity
    sim = torch.mm(test_features_skt, test_features_img.T)
    k = {'map': test_features_skt.shape[0], 'precision': 100}
    ############################################################################
    # Step 3: evaluate
    aps = map_sake(test_features_img.numpy(), gallery_labels.numpy(),
                   test_features_skt.numpy(), gallery_labels_skt.numpy(),sim, k=k['map'])
    prec = prec_sake(test_features_img.numpy(), gallery_labels.numpy(),
                   test_features_skt.numpy(), gallery_labels_skt.numpy(),sim,k=k['precision'])
    print('map{}: {:.4f} prec{}: {:.4f}'.format(k['map'], np.mean(aps), k['precision'], prec))

def evaluate_student(args):
    datasets, sk_valid_data, im_valid_data = load_data(args)
    model_skt = vits.Student_SKT(datasets.get_num_class(),checkpoint_path='vit.npz').cuda()
    model_img = vits.Student_IMG(datasets.get_num_class(),checkpoint_path='vit.npz').cuda()
    model_skt.load_state_dict(torch.load(f'weights/student_{args.dataset}_skt_good.pth'))
    model_img.load_state_dict(torch.load(f'weights/student_{args.dataset}_img_good.pth'))

    skt_loader = DataLoader(sk_valid_data, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)
    img_loader = DataLoader(im_valid_data, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)
    print(mAP1(img_loader,skt_loader,model_img,model_skt))

def valid(skt_loader, img_loader,model_img,model_skt):
    model_img.eval()
    model_skt.eval()
    acc= mAPcuda(img_loader, skt_loader, model_img,model_skt, k=100)
    model_img.train()
    model_skt.train()
    return acc

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
        parser.add_argument('--testall', default=True, action='store_true', help='train/test scale')
        parser.add_argument("--seed", default=0)
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

if __name__ == '__main__':
    args=Option().parse()
    evaluate_student(args)
