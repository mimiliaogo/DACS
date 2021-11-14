import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SynthiaDataSet
from data.nthu_husky_loader import NTHU_HUSKYLoader
from data.unity_dataset import UnityDataSet

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSet,
        "nthu_husky": NTHU_HUSKYLoader,
        "unity": UnityDataSet
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '/work/engine210/Dataset/cityscapes/'
    if name == 'gta' or name == 'gtaUniform':
        return '/work/engine210/Dataset/GTA5/'
    if name == 'synthia':
        return '../data/RAND_CITYSCAPES'
    if name == 'nthu_husky':
        return '/work/engine210/Dataset/NTHU_HUSKY/library_to_EECS'
    if name == 'unity':
        return '/work/engine210/Dataset/NTHU_HUSKY/UNITY_DATASET'
