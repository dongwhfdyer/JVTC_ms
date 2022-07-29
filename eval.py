import os, torch
import numpy as np
from scipy.spatial.distance import cdist

from config.resnet_config import config
from dataset import imgdataset, imgdataset_camtrans, create_dataset
from utils.util import cluster, get_info
from utils.util import extract_fea_camtrans, extract_fea_test
from utils.st_distribution import get_st_distribution
from resnet import ResNet, load_ms_resnet50_model, Bottleneck
import mindspore.nn as nn

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ################################################## params
    dataset_path = 'data'
    ann_file_train = 'list_market/list_market_train.txt'
    ann_file_test = 'list_market/list_market_test.txt'

    snapshot = 'evalution/resnet50_duke2market_epoch00100.pth'

    num_cam = 6
    ##################################################
    img_dir = dataset_path + '/market_merge'
    train_dataset = create_dataset(dataset_dir=img_dir, ann_file=ann_file_train, batch_size=1, state='test')

    img_dir = dataset_path + '/Market-1501-v15.09.15/Market-1501-v15.09.15'
    test_dataset = imgdataset(dataset_dir=img_dir, txt_path=ann_file_test, transformer='test')

    ################################################## kuhn test code
    # for i in range(10):
    #     data_item, _, _ = test_dataset.__getitem__(i)
    #     get_statistics(data_item)
    # exit()
    ##################################################

    model = ResNet(Bottleneck, [3, 4, 6, 3], config.class_num, train=False)
    model, _ = load_ms_resnet50_model(model, snapshot)
    model.set_train(False)

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
    evaluate_joint(test_fea=test_feas, st_distribute=distribution, ann_file=ann_file_test, select_set='market')
