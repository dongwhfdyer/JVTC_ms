import os, torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from scipy.spatial.distance import cdist

from utils.util import cluster, get_info
from utils.util import extract_fea_camtrans, extract_fea_test
from utils.resnet import resnet50
from utils.dataset import imgdataset, imgdataset_camtrans
from utils.rerank import re_ranking
from utils.st_distribution import get_st_distribution
from utils.evaluate_joint_sim import evaluate_joint
import mindspore.nn as nn

if __name__ == '__main__':
    pass
