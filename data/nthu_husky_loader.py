import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data
from PIL import Image

from data.city_utils import recursive_glob
from data.augmentations import *
from utils.randaugment import RandAugmentMC

class NTHU_HUSKYLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {"cityscapes": [73.15835921, 82.90891754, 72.39239876],}

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        img_norm=False,
        augmentations=None,
        #[TODO:mimi]
        strong_augmentations=True,
        version="cityscapes",
        return_id=False,
        img_mean = np.array([73.15835921, 82.90891754, 72.39239876])
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        # [TODO: mimi] strong aug
        self.strong_augmentations = strong_augmentations
        self.randaug = RandAugmentMC(4, 10) # make augmentation stronger
        self.img_norm = img_norm
        self.n_classes = 9
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array([128, 128, 128])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        # self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        # TODO: nthu
        list_path = '/home/engine210/mimi/DACS/data/nthu_husky_list/train.txt'
        self.files["train"] = []
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        for name in self.img_ids:
            img_file = os.path.join(self.root, name)
            self.files[self.split].append(img_file)

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

        self.return_id = return_id

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        # lbl_path = os.path.join(
        #     self.annotations_base,
        #     img_path.split(os.sep)[-2], # temporary for cross validation
        #     os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        # )
        # TODO: dummy label path
        lbl_path = '/work/engine210/Dataset/GTA5/labels/07136.png'

        # img = m.imread(img_path)
        # img = np.array(img, dtype=np.uint8)
        # lbl = m.imread(lbl_path)
        # lbl = np.array(lbl, dtype=np.uint8)
        # lbl = self.encode_segmap(lbl)

        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)

        # resize
        img = img.resize(self.img_size, Image.BICUBIC)
        lbl = lbl.resize(self.img_size, Image.NEAREST)

        img = np.asarray(img, np.uint8)
        lbl = np.asarray(lbl, np.uint8)


        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        # img = np.asarray(img, np.float32)
        # lbl = np.asarray(lbl, np.float32)

        # [TODO:mimi] strong augmentation
        img_strong = img
        params_strong = None
        if self.strong_augmentations:
            img_strong, params_strong = self.randaug(Image.fromarray(img))
        
        # if self.is_transform:
        #         img_strong, _ = self.transform(img_strong, lbl)
        #         img, lbl = self.transform(img, lbl)

        img = np.asarray(img, np.float32)
        img_strong = np.asarray(img_strong, np.float32)
        lbl = np.asarray(lbl, np.float32)

        img = img[:, :, ::-1]  # change to BGR
        img -= self.mean
        img = img.transpose((2, 0, 1))

        img_strong = img_strong[:, :, ::-1]  # change to BGR
        img_strong -= self.mean
        img_strong = img_strong.transpose((2, 0, 1))

        img_name = img_path.split('/')[-1]
        if self.return_id:
            return img, lbl, img_name, img_name, index


        return img.copy(), img_strong.copy(), params_strong, lbl.copy(), img_path, lbl_path, img_name


'''
if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip()])

    local_path = "./data/city_dataset/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
'''
