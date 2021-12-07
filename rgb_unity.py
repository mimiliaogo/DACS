from PIL import Image
import numpy as np

rgb_to_trainid = {
            (128, 64, 128): 0,
            (244, 35, 232): 1,
            (70, 70, 70): 2,
            (107, 142, 35): 3,
            (152, 251, 152): 4,
            (0, 0, 0): 5,
            (220, 20, 60): 6,
            (0, 0, 142): 7,
            (119, 11, 32): 8,
            (255, 255, 0): 9
}

rgb_b_to_trainid = {
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
id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 2, 13: 2, 21: 3, 
                22: 4, 23: 5, 24: 6, 25: 6, 26: 7, 27: 7, 28: 7,
                31: 7, 32: 8, 33: 8}

img_pth = '/home/engine210/Dataset/NTHU_HUSKY/Unity_Dataset_noObstacles/202111201959_00328_seg.png'
lbl_path = '/work/engine210/Dataset/GTA5/labels/07136.png'

label = Image.open(img_pth)
label = label.resize((1280,720), Image.NEAREST)
label = np.asarray(label, np.uint8)

# obtain rgb b channel
label_g = label[:, :, 2]

label_copy = 255 * np.ones((720, 1280), dtype=np.float32)
for k, v in rgb_b_to_trainid.items():
    label_copy[label_g == k] = v

# remove obstalcle
label_r = label[:, :, 1]
label_copy[label_r == 255] = 255
print(label_copy.shape)
print(np.unique(label_copy))
from utils.helpers import colorize_mask

NTHU_HUSKY_palette =  [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    107, 142, 35,
    152, 251, 152,
    70, 130, 180,
    220, 20, 60,
    0, 0, 142,
    119, 11, 32,
    255, 255, 0
]

colorized_mask = colorize_mask(label_copy, NTHU_HUSKY_palette)
colorized_mask.save('unity_pred.png')
