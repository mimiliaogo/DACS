import argparse
import cv2
# from skimage import feature
import numpy as np
# np.set_printoptions(threshold=np.inf)
import sys
from collections import OrderedDict
import os
from numpy.core.numeric import full

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from model import deeplabv2
from model import deeplabv3
from data import get_data_path, get_loader
# from utils.visualization import save_image

from PIL import Image
import timeit

class DACS():
    def __init__(self):
        self.mean = np.array([96.2056, 101.4815, 100.8839])
        self.label_size = [720, 1280]
        # mdoel config
        # dacs
        # self.ckpt_path = '/home/engine210/mimi/DACS/saved/DeepLabv2-nthu/10-21_14-48-UDA-gta-nthu_resume-10-23_23-45/checkpoint-iter204000.pth'
        # rainbow
        self.ckpt_path = '/home/engine210/mimi/end-uda/nthu_deeplabv3+/35000_model.pth'
        # src only
        self.ckpt_path = '/home/engine210/mimi/DACS/saved/DeepLabv2-nthu-unity/11-22_11-50-nthu-unity-noObstacle/checkpoint-iter110000.pth'

        self.num_classes = 9
        
        self.model = deeplabv2.Res_Deeplab(num_classes=self.num_classes)
        # self.model = deeplabv3.Deeplabv3(backbone='mobilenet', num_classes=9)
        checkpoint = torch.load(self.ckpt_path)
        try: #dacs
            self.model.load_state_dict(checkpoint['model'])
        except:
            # rainbow
            self.model.load_state_dict(checkpoint)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        print('device:', self.device)
        self.model = self.model.cuda()
        self.model.eval()

        colors = [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [0, 0, 142],
            [119, 11, 32],
        ]

        class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "vegetation",
            "terrain",
            "sky",
            "person/rider",
            "car/truck/bus",
            "motorcycle/bicycle",
        ]

        self.label_colours = dict(zip(range(10), colors))

    def inference(self, img):
        img = np.asarray(img, np.float32)
        img = img[:, :, ::-1]  # change to BGR
        img -= self.mean
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img.copy()).float()
        interp = nn.Upsample(size=(720,1280), mode='bilinear', align_corners=True)

        
        with torch.no_grad():
            outputs = self.model(Variable(img).cuda())
            output = interp(outputs["S"]).squeeze()
            output = output.cpu().data.numpy()
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
            output = self.decode_segmap(output)
        
        return output

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, 9):
            r[temp == l] = self.label_colours[l][2]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][0]
        r[temp == 255] = 0
        g[temp == 255] = 0
        b[temp == 255] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    def get_iou(self, data_list, class_num, dataset, save_path=None):
        from multiprocessing import Pool
        from utils.metric import ConfusionMatrix

        ConfM = ConfusionMatrix(class_num)
        f = ConfM.generateM
        pool = Pool()
        m_list = pool.map(f, data_list)
        pool.close()
        pool.join()

        for m in m_list:
            ConfM.addM(m)

        aveJ, j_list, M = ConfM.jaccard()

        classes = np.array(("road", "sidewalk",
            "building", "vegetation",
            "terrain", "sky", "person", "car", "bike"))


        for i, iou in enumerate(j_list):
            print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]))

        print('meanIOU: ' + str(aveJ) + '\n')
        if save_path:
            with open(save_path, 'w') as f:
                for i, iou in enumerate(j_list):
                    f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]) + '\n')
                f.write('meanIOU: ' + str(aveJ) + '\n')
        return aveJ, j_list


if __name__ == "__main__":
    model = DACS()
    data_loader = get_loader('nthu_husky')
    list_path = '/home/engine210/Dataset/NTHU_HUSKY/val.txt'
    test_dataset = data_loader('/work/engine210/Dataset/NTHU_HUSKY/label', img_size=(1280,720), is_transform=True, split='val', list_path = list_path)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    interp = nn.Upsample(size=(720,1280), mode='bilinear', align_corners=True)
    ignore_label = 255
    print('Evaluating, found ' + str(len(testloader)) + ' batches.')
    n_saved = 0
    data_list = []

    for index, batch in enumerate(testloader):
        # imgs, labels, size, name, full_name = batch
        imgs, _, _, labels, img_pth, lbl_pth, name = batch
        with torch.no_grad():
            outputs  = model.model(Variable(imgs).cuda())
            # if isinstance(outputs, dict):
            #     depths = torch.sigmoid(interp(outputs["D"]))
            #     outputs = interp(outputs["S"])
            outputs = interp(outputs)

            for image, output, label, name in zip(imgs,outputs, labels, img_pth):
                # loss = criterion(output.unsqueeze(0), label_cuda.unsqueeze(0))
                # total_loss.append(loss.item())

                output = output.cpu().data.numpy()
                # depth = depth.cpu().data.numpy()
                gt = np.asarray(label.numpy(), dtype=np.int)
                print(np.unique(gt))

                output = output.transpose(1,2,0)
                # depth = depth.transpose(1,2,0).squeeze()*255
                # assert False

                output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

                # save error map
                # error_map = np.zeros(output.shape[:2])
                # error_map[output != gt] = 255
                # error_map[gt == 250] = 0 # unlabel index
                # cv2.imwrite(f'./eval_img/{i}_errormap.png', error_map)

                out_dir = './eval_rainbow_NTHU/'
                output_save = model.decode_segmap(output)
                img = cv2.imread(name)
                cv2.imwrite(f'{out_dir}/{n_saved}_0img.png', img)
                cv2.imwrite(f'{out_dir}/{n_saved}_1pred.png', output_save)
                cv2.imwrite(f'{out_dir}/{n_saved}_2gt.png', model.decode_segmap(gt))
                # save_image(out_dir, gt, None, f'{n_saved}_2gt')
                print(n_saved)
                n_saved += 1
                data_list.append([gt.flatten(), output.flatten()])

        # if (index+1) % 10 == 0:
        #     print('%d processed'%(index+1))

    mIoU, cIoU = model.get_iou(data_list, 9, 'nthu', './eval_rainbow_NTHU/result.txt')
    print('mIoU: %f'%(mIoU))
    
    # start = timeit.default_timer()
    # for i in range(10):
    #     mask = model.inference(img)
    #     end = timeit.default_timer()
    #     print('Total time: ' + str(end-start) + 'seconds')
    # cv2.imwrite('./test.png', mask)



