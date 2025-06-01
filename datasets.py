# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 16:34
# @Author : 李昌杏
# @File : datasets.py
# @Software : PyCharm

import argparse

import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch.utils import data
from torchvision.transforms import transforms

def get_pic(path):
    return (Image.fromarray(np.array(Image.open(path).convert('RGB'))))

# 预加载的一些文件
def load_para(args):
    # test class labels
    if args.dataset == 'Sketchy21' or args.dataset == 'Sketchy25':
        if args.test_class == 'test_class_sketchy25':
            with open(args.data_path + "/Sketchy/zeroshot1/cname_cid_zero.txt", 'r') as f:
                file_content = f.readlines()
                test_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
            train_dir = args.data_path + "/Sketchy/zeroshot1/cname_cid.txt"
            with open(train_dir, 'r') as f:
                file_content = f.readlines()
                train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

        elif args.test_class == "test_class_sketchy21":
            with open(args.data_path + "/Sketchy/zeroshot0/cname_cid_zero.txt", 'r') as f:
                file_content = f.readlines()
                test_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
            train_dir = args.data_path + "/Sketchy/zeroshot0/cname_cid.txt"
            with open(train_dir, 'r') as f:
                file_content = f.readlines()
                train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

    elif args.dataset == 'tu_berlin':
        if args.test_class == 'test_class_tuberlin30':
            with open(args.data_path + "/TUBerlin/zeroshot/cname_cid_zero.txt", 'r') as f:
                file_content = f.readlines()
                test_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
            train_dir = args.data_path + "/TUBerlin/zeroshot/cname_cid.txt"
            with open(train_dir, 'r') as f:
                file_content = f.readlines()
                train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

    elif args.dataset == 'Quickdraw':
        with open(args.data_path + "/QuickDraw/zeroshot/cname_cid_zero.txt", 'r') as f:
            file_content = f.readlines()
            test_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        train_dir = args.data_path + "/QuickDraw/zeroshot/cname_cid.txt"
        with open(train_dir, 'r') as f:
            file_content = f.readlines()
            train_class_label = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])

    print('training classes: ', train_class_label.shape)
    print('testing classes: ', test_class_label.shape)
    return train_class_label, test_class_label

class PreLoad:
    def __init__(self, args):
        self.all_valid_or_test_sketch = []
        self.all_valid_or_test_sketch_label = []
        self.all_valid_or_test_image = []
        self.all_valid_or_test_image_label = []

        self.all_train_sketch = []
        self.all_train_sketch_label = []
        self.all_train_image = []
        self.all_train_image_label = []

        self.all_train_sketch_cls_name = []
        self.all_train_image_cls_name = []

        self.init_valid_or_test(args)
        # load_para(args)

    def init_valid_or_test(self, args):
        if args.dataset == 'Sketchy21' or args.dataset == 'Sketchy25'  :
            train_dir = args.data_path + '/Sketchy/'
        elif args.dataset == 'tu_berlin':
            train_dir = args.data_path + '/TUBerlin/'
        elif args.dataset == 'Quickdraw':
            train_dir = args.data_path + '/QuickDraw/'
        else:
            NameError("Dataset is not implemented")

        self.all_valid_or_test_sketch, self.all_valid_or_test_sketch_label = \
            get_file_list_iccv(args, train_dir, "sketch", "test")
        self.all_valid_or_test_image, self.all_valid_or_test_image_label = \
            get_file_list_iccv(args, train_dir, "images", "test")

        self.all_train_sketch, self.all_train_sketch_label, self.all_train_sketch_cls_name =\
            get_all_train_file(args, "sketch")
        self.all_train_image, self.all_train_image_label, self.all_train_image_cls_name = \
            get_all_train_file(args, "image")

        print("used for valid or test sketch / image:")
        print(self.all_valid_or_test_sketch.shape, self.all_valid_or_test_image.shape)
        print("used for train sketch / image:")
        print(self.all_train_sketch.shape, self.all_train_image.shape)

def get_all_train_file(args, skim):
    if skim != 'sketch' or skim != 'image':
        NameError(skim + ' not implemented!')

    if args.dataset == 'Sketchy21' or args.dataset == 'Sketchy25':
        if args.test_class == "test_class_sketchy25":
            shot_dir = "zeroshot1"
        elif args.test_class == "test_class_sketchy21":
            shot_dir = "zeroshot0"

        cname_cid = args.data_path + f'/Sketchy/{shot_dir}/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_train.txt'
        elif skim == 'image':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/all_photo_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif args.dataset == 'tu_berlin':
        cname_cid = args.data_path + '/TUBerlin/zeroshot/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/png_ready_filelist_train.txt'
        elif skim == 'image':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/ImageResized_ready_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif args.dataset == 'Quickdraw':
        cname_cid = args.data_path + '/QuickDraw/zeroshot/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + '/QuickDraw/zeroshot/sketch_train.txt'
        elif skim == 'image':
            file_ls_file = args.data_path + '/QuickDraw/zeroshot/all_photo_train.txt'
        else:
            NameError(skim + ' not implemented!')

    else:
        NameError(skim + ' not implemented! ')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()

    # 图片相对路径
    file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    # 图片的label,0,1,2...
    labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])

    # 所有的训练类
    with open(cname_cid, 'r') as ci:
        class_and_indx = ci.readlines()
    # 类名
    cname = np.array([' '.join(cc.strip().split()[:-1]) for cc in class_and_indx])

    return file_ls, labels, cname

def get_file_iccv(labels, rootpath, class_name, cname, number, file_ls):
    # 该类的label
    label = np.argwhere(cname == class_name)[0, 0]

    # 该类的所有样本
    ind = np.argwhere(labels == label)
    ind_rand = np.random.randint(1, len(ind), number)
    ind_ori = ind[ind_rand]
    files = file_ls[ind_ori][0][0]
    full_path = rootpath+"/"+ files
    return full_path

def get_file_list_iccv(args, rootpath, skim, split):

    if args.dataset == 'Sketchy21' or args.dataset == 'Sketchy25':
        if args.test_class == "test_class_sketchy25":
            shot_dir = "zeroshot1"
        elif args.test_class == "test_class_sketchy21":
            shot_dir = "zeroshot0"
        else:
            NameError("zeroshot is invalid")

        if skim == 'sketch':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/all_photo_filelist_zero.txt'

    elif args.dataset == 'tu_berlin':
        if skim == 'sketch':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/png_ready_filelist_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/ImageResized_ready_filelist_zero.txt'

    elif args.dataset == 'Quickdraw':
        if skim == 'sketch':
            file_ls_file = args.data_path + f'/QuickDraw/zeroshot/sketch_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + f'/QuickDraw/zeroshot/all_photo_zero.txt'

    else:
        NameError(args.dataset + 'is invalid')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()
    file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
    file_names = np.array([(rootpath + x) for x in file_ls])

    # 对验证的样本数量进行缩减
    # sketch 15229->762 image 17101->1711
    if (args.dataset == 'Sketchy21' or args.dataset == 'Sketchy25') and split == 'test' and skim == 'sketch':
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 15229
        else:
            index = [i for i in range(0, file_names.shape[0], 20)]   # 762
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == 'sketchy_extend' and split == 'test' and skim == 'images':
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 17101
        else:
            index = [i for i in range(0, file_names.shape[0], 10)]  # 1711
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    # sketch 2400->800, image 27989->1400
    if args.dataset == "tu_berlin" and skim == "sketch" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 2400
        else:
            index = [i for i in range(0, file_names.shape[0], 3)]  # 800
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == "tu_berlin" and skim == "images" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 27989
        else:
            index = [i for i in range(0, file_names.shape[0], 20)]  # 1400
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    # Quickdraw 92291->770, image 54151->1806
    if args.dataset == "Quickdraw" and skim == "sketch" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 92291
        else:
            index = [i for i in range(0, file_names.shape[0], 120)]  # 770
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == "Quickdraw" and skim == "images" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 54151
        else:
            index = [i for i in range(0, file_names.shape[0], 30)]  # 1806
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    file_names_cls = labels
    return file_names, file_names_cls

def remove_white_space_image(img_np: np.ndarray, padding: int):
    """
    获取白底图片中, 物体的bbox; 此处白底必须是纯白色.
    其中, 白底有两种表示方法, 分别是 1.0 以及 255; 在开始时进行检查并且匹配
    对最大值为255的图片进行操作.
    三通道的图无法直接使用255进行操作, 为了减小计算, 直接将三通道相加, 值为255*3的pix 认为是白底.
    :param img_np:
    :return:
    """
    # if np.max(img_np) <= 1.0:  # 1.0 <= 1.0 True
    #     img_np = (img_np * 255).astype("uint8")
    # else:
    #     img_np = img_np.astype("uint8")

    h, w, c = img_np.shape
    img_np_single = np.sum(img_np, axis=2)
    Y, X = np.where(img_np_single <= 300)  # max = 300
    ymin, ymax, xmin, xmax = np.min(Y), np.max(Y), np.min(X), np.max(X)
    img_cropped = img_np[max(0, ymin - padding):min(h, ymax + padding), max(0, xmin - padding):min(w, xmax + padding),
                  :]
    return img_cropped

def resize_image_by_ratio(img_np: np.ndarray, size: int):
    """
    按照比例resize
    :param img_np:
    :param size:
    :return:
    """
    # print(len(img_np.shape))
    if len(img_np.shape) == 2:
        h, w = img_np.shape
    elif len(img_np.shape) == 3:
        h, w, _ = img_np.shape
    else:
        assert 0

    ratio = h / w
    if h > w:
        new_img = cv2.resize(img_np, (int(size / ratio), size,))  # resize is w, h  (fx, fy...)
    else:
        new_img = cv2.resize(img_np, (size, int(size * ratio),))
    # new_img[np.where(new_img < 200)] = 0
    return new_img

def make_img_square(img_np: np.ndarray):
    if len(img_np.shape) == 2:
        h, w = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1)) * np.max(img_np)
            white2 = np.ones((h, delta2)) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w)) * np.max(img_np)
            white2 = np.ones((delta2, w)) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        if h > w:
            delta1 = (h - w) // 2
            delta2 = (h - w) - delta1

            white1 = np.ones((h, delta1, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((h, delta2, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.hstack([white1, img_np, white2])
            return new_img
        else:
            delta1 = (w - h) // 2
            delta2 = (w - h) - delta1

            white1 = np.ones((delta1, w, c), dtype=img_np.dtype) * np.max(img_np)
            white2 = np.ones((delta2, w, c), dtype=img_np.dtype) * np.max(img_np)

            new_img = np.vstack([white1, img_np, white2])
            return new_img

# 每个label，对应一个数字
def create_dict_texts(texts):
    texts = list(texts)
    dicts = {l: i for i, l in enumerate(texts)}
    return dicts

def preprocess(image_path, img_type="im"):
    # immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    # imstd = [0.229, 0.224, 0.225]

    immean = [0.5, 0.5, 0.5]  # RGB channel mean for imagenet
    imstd = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(immean, imstd),
    ])

    # cv2.imshow(f'{image_path}',cv2.imread(image_path))
    # cv2.waitKey(0)

    if img_type == 'im':
        orgi=Image.open(image_path).resize((224, 224)).convert('RGB')
        return transform(orgi),cv2.imread(image_path)
    else:
        # 对sketch 进行crop，等比例扩大到224
        orgi = cv2.imread(image_path)
        img = cv2.cvtColor(orgi, cv2.COLOR_BGR2RGB)
        img = remove_white_space_image(img, 10)
        img = resize_image_by_ratio(img, 224)
        img = make_img_square(img)
        return transform(img),orgi

def load_data_test(args):
    pre_load = PreLoad(args)
    sk_valid_data = ValidSet(pre_load, 'sk', half=True)
    im_valid_data = ValidSet(pre_load, 'im', half=True)
    return sk_valid_data, im_valid_data

def load_data(args,org=False):
    train_class_label, test_class_label = load_para(args)  # cls : 类名
    pre_load = PreLoad(args)
    train_data = TrainSet(args, train_class_label, pre_load)
    sk_valid_data = ValidSet(pre_load, 'sk',org=org)
    im_valid_data = ValidSet(pre_load, 'im',org=org)
    return train_data, sk_valid_data, im_valid_data

class TrainSet(data.Dataset):
    def __init__(self, args, train_class_label, pre_load):
        self.args = args
        self.pre_load = pre_load
        self.train_class_label = train_class_label
        self.class_dict = create_dict_texts(train_class_label)
        if self.args.dataset == 'Sketchy21' or args.dataset == 'Sketchy25':
            self.root_dir =  args.data_path + '/Sketchy'
        elif self.args.dataset == 'tu_berlin':
            self.root_dir = args.data_path + '/TUBerlin'
        elif self.args.dataset == 'Quickdraw':
            self.root_dir = args.data_path + '/QuickDraw'

    def __getitem__(self, index):
        # choose 3 label
        self.choose_label_name = np.random.choice(self.train_class_label, 3, replace=False)

        label = self.class_dict.get(self.choose_label_name[0])
        label_neg = self.class_dict.get(self.choose_label_name[-1])

        sketch = get_file_iccv(self.pre_load.all_train_sketch_label, self.root_dir, self.choose_label_name[0],
                               self.pre_load.all_train_sketch_cls_name, 1, self.pre_load.all_train_sketch)
        image = get_file_iccv(self.pre_load.all_train_image_label, self.root_dir, self.choose_label_name[0],
                              self.pre_load.all_train_image_cls_name, 1, self.pre_load.all_train_image)
        sketch_neg = get_file_iccv(self.pre_load.all_train_sketch_label, self.root_dir, self.choose_label_name[-1],
                                   self.pre_load.all_train_sketch_cls_name, 1, self.pre_load.all_train_sketch)
        image_neg = get_file_iccv(self.pre_load.all_train_image_label, self.root_dir, self.choose_label_name[-1],
                                  self.pre_load.all_train_image_cls_name, 1, self.pre_load.all_train_image)
        # print(sketch,image,sketch_neg,image_neg)
        sketch = preprocess(sketch, 'sk')[0]
        image = preprocess(image)[0]
        sketch_neg = preprocess(sketch_neg, 'sk')[0]
        image_neg = preprocess(image_neg)[0]
        return sketch, image, sketch_neg, image_neg,label,label_neg

    def __len__(self):
        return self.args.dataset_len

    def get_num_class(self):return len(self.train_class_label)

class ValidSet(data.Dataset):

    def __init__(self, pre_load, type_skim='im', half=False, path=False,org=False):
        self.type_skim = type_skim
        self.half = half
        self.path = path
        if type_skim == "sk":
            self.file_names, self.cls = pre_load.all_valid_or_test_sketch, pre_load.all_valid_or_test_sketch_label
        elif type_skim == "im":
            self.file_names, self.cls = pre_load.all_valid_or_test_image, pre_load.all_valid_or_test_image_label
        else:
            NameError(type_skim + " is not right")
        self.org=org

    def __getitem__(self, index):
        label = self.cls[index]  # label 为数字
        file_name = self.file_names[index]
        if self.path:
            image = file_name
        else:
            if self.half:
                image,orgi = preprocess(file_name, self.type_skim)
            else:
                image,orgi = preprocess(file_name, self.type_skim)
        if self.org:return image, label,orgi
        return image, label

    def __len__(self):
        return len(self.file_names)

class Option:

    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")

        # dataset
        parser.add_argument('--data_path', type=str, default="../datasets")
        parser.add_argument('--dataset', type=str, default='Sketchy21',
                            choices=['Sketchy21', 'tu_berlin', 'Quickdraw'])
        parser.add_argument('--test_class', type=str, default='test_class_sketchy21',
                            choices=['test_class_sketchy25', 'test_class_sketchy21', 'test_class_tuberlin30', 'Quickdraw'])
        parser.add_argument('--testall', default=True, action='store_true', help='train/test scale')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()