import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class UnityDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, augmentations = None, img_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=250):
        self.root = root
        self.list_path = list_path
        self.img_size = img_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.augmentations = augmentations
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        # self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        #                       19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
        #                       26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        # TODO: nthu (9 classes)
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 2, 13: 2, 21: 3, 
                                22: 4, 23: 5, 24: 6, 25: 6, 26: 7, 27: 7, 28: 7,
                                31: 7, 32: 8, 33: 8}


        self.rgb_b_to_trainid = {
            128: 0,
            232: 1,
            70: 2,
            35: 3,
            152: 4,
            0: 5,
            60: 6,
            142: 7,
            32: 8
        }
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            # TODO : path are all included in train list
            # img_file = osp.join(self.root, "images/%s" % name)
            # label_file = osp.join(self.root, "labels/%s" % name)
            img_file = "%s_screenshot.png" % name
            label_file = "%s_seg.png" % name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.img_size, Image.BICUBIC)
        label = label.resize(self.img_size, Image.NEAREST)

        image = np.asarray(image, np.uint8)
        label = np.asarray(label, np.uint8)

        # obtain rgb b channel
        label_b = label[:, :, 2]

        label_copy = 255 * np.ones((720, 1280), dtype=np.float32)
        for k, v in self.rgb_b_to_trainid.items():
            label_copy[label_b == k] = v

        # remove obstalcle
        label_g = label[:, :, 1]
        label_copy[label_g == 255] = 255
        print("before aug")
        print(np.unique(label_copy)) 
        if self.augmentations is not None:
            image, label_copy = self.augmentations(image, Image.fromarray(label_copy))

        print(np.unique(label_copy)) 
        image = np.asarray(image, np.float32)
        label_copy = np.asarray(label_copy, np.float32)       
        print(label_copy.shape)
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = UnityDataSet("/home/engine210/Dataset/NTHU_HUSKY/UNITY_DATASET", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
