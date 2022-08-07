import os, torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from scipy.spatial.distance import cdist
import numpy as np
from mindspore import context
from scipy.spatial.distance import cdist

from config.resnet_config import config
from dataset import create_dataset
from resnet import ResNet, load_ms_resnet50_model, Bottleneck
from st_distribution import get_st_distribution
from utils import extract_fea_camtrans, get_info, extract_fea_test, cluster

dataset_path = 'data'
##########nhuk####################################
ann_file_train = 'list_duke/list_duke_train.txt'
ann_file_test = 'list_duke/list_duke_test.txt'
##########nhuk####################################

snapshot = 'evalution/resnet50_market2duke_epoch00100.pth'

num_cam = 8
###########   DATASET   ###########
img_dir = dataset_path + 'duke/bounding_box_train_camstyle_merge/'
train_dataset = imgdataset_camtrans(dataset_dir=img_dir, txt_path=ann_file_train,
                                    transformer='test', K=num_cam, num_cam=num_cam)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=4)

img_dir = dataset_path + 'duke/'
test_dataset = imgdataset(dataset_dir=img_dir, txt_path=ann_file_test, transformer='test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

###########   TEST   ###########
model, _ = resnet50(pretrained=snapshot, num_classes=751)
model.cuda()
model.eval()

print('extract feature for training set')
train_feas = extract_fea_camtrans(model, train_loader)
_, cam_ids, frames = get_info(ann_file_train)

print('generate spatial-temporal distribution')
dist = cdist(train_feas, train_feas)
dist = np.power(dist, 2)
# dist = re_ranking(original_dist=dist)
labels = cluster(dist)
num_ids = len(set(labels))
print('cluster id num:', num_ids)
distribution = get_st_distribution(cam_ids, labels, frames, id_num=num_ids, cam_num=num_cam)

print('extract feature for testing set')
test_feas = extract_fea_test(model, test_loader)

print('evaluation')
evaluate_joint(test_fea=test_feas, st_distribute=distribution, ann_file=ann_file_test, select_set='duke')
################################################## params
if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    dataset_path = 'data'
    ann_file_train = 'list_duke/list_duke_train.txt'
    ann_file_test = 'list_duke/list_duke_test.txt'

    snapshot = 'm_resnet.ckpt'  # todo

    num_cam = 8
    ##################################################
    train_dataset_path = dataset_path + '/market_merge'
    test_dataset_path = dataset_path + '/Market-1501-v15.09.15/Market-1501-v15.09.15'
    train_dataset = create_dataset(dataset_dir=train_dataset_path, ann_file=ann_file_train, batch_size=1, state='train', num_cam=num_cam, K=num_cam)

    test_dataset = create_dataset(dataset_dir=test_dataset_path, ann_file=ann_file_test, state='test', batch_size=1)
