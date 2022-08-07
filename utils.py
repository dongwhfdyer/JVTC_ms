import pdb
from sklearn.cluster import DBSCAN

import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P


def l2_dist(fea_query, fea_gallery):
    dist = np.zeros((fea_query.shape[0], fea_gallery.shape[0]), dtype=np.float64)
    for i in range(fea_query.shape[0]):
        dist[i, :] = np.sum((fea_gallery - fea_query[i, :]) ** 2, axis=1)
    return dist


def get_info(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        # self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in lines]
        labels = [int(i.split()[1]) for i in lines]
        cam_ids = [int(i.split()[2]) for i in lines]
        frames = [int(i.split()[3]) for i in lines]

    return labels, cam_ids, frames


def cluster(dist, rho=1.6e-3):
    tri_mat = np.triu(dist, 1)
    tri_mat = tri_mat[np.nonzero(tri_mat)]
    tri_mat = np.sort(tri_mat, axis=None)
    top_num = np.round(rho * tri_mat.size).astype(int)
    eps = tri_mat[:top_num].mean()  # *2
    # print('eps in cluster: {:.3f}'.format(eps))
    cluster = DBSCAN(eps=eps, min_samples=1, metric='precomputed', n_jobs=8)
    labels = cluster.fit_predict(dist)

    return labels


def concat_using_numpy(ms_tensor):
    temp = np.concatenate([ms_tensor[i].asnumpy() for i in range(len(ms_tensor))], axis=0)
    # temp = np.concatenate([ms_tensor[i].asnumpy() for i in range(ms_tensor.size())], axis=0)
    ms_tensor_concat = Tensor(temp)
    return ms_tensor_concat


def extract_fea_camtrans(model, loader, num_cam=6, K=6):
    feas = []
    batch_size = 1
    for data in loader.create_dict_iterator():
        # columns_names_list = ['images' + str(i) for i in range(K)]

        # for ind in columns_names_list:
        #     data[ind] = P.ExpandDims()(data[ind], 0)
        # waiting_concat_ = (data[key] for key in columns_names_list)

        # cconcat_ = P.Concat()

        # numpy_data = [data[key].asnumpy() for key in columns_names_list]
        # concat_data = np.concatenate(numpy_data, axis=0)
        # concat_images = Tensor(concat_data)

        concat_images_ = [data['images' + str(i)] for i in range(num_cam)]

        concat_images = P.Concat()(concat_images_)
        # concat_images = P.Concat()((data['images0'], data['images1'], data['images2'], data['images3'], data['images4'], data['images5']))

        np.save("rubb/ms_concat_images.npy", concat_images.asnumpy())  # todo: delete
        out = model(concat_images)
        np.save("rubb/ms_out_0.npy", out[0].asnumpy())
        np.save("rubb/ms_out_1.npy", out[1].asnumpy())
        fea = out[2]
        fea = fea.view(batch_size, K, -1)
        fea = fea.mean(axis=1)
        np.save("rubb/ms_fea_mean.npy", fea.asnumpy())
        fea = P.L2Normalize(axis=1, epsilon=1e-12)(fea)  # kuhn: important difference
        np.save("rubb/ms_fea.npy", fea.asnumpy())
        feas.append(fea)

    feas = P.Concat()(feas)
    # feas = concat_using_numpy(feas)  # kuhn edted

    return feas.asnumpy()


# def extract_fea_camtrans(model, loader):
#     feas = []
#     for i, data in enumerate(loader, 1):
#         # break
#         with torch.no_grad():
#             image = data[0].cuda()
#
#             batch_size = image.size(0)
#             K = image.size(1)
#
#             image = image.view(image.size(0) * image.size(1), image.size(2), image.size(3), image.size(4))
#             # image = Variable(image).cuda()
#             out = model(image)
#             fea = out[2]
#             fea = fea.view(batch_size, K, -1)
#             fea = fea.mean(dim=1)
#             fea = F.normalize(fea)
#             feas.append(fea)
#
#     feas = torch.cat(feas)
#     # print('duke_train_feas', feas.size())
#     return feas.cpu().numpy()


def extract_fea_test(model, loader):
    feas = []
    for data in loader.create_dict_iterator():
        image = data['images']
        out = model(image)
        fea = out[1]
        feas.append(fea)

    feas = P.Concat()(feas)
    print("feas", feas.shape)
    return feas.asnumpy()

    # for i, data in enumerate(loader, 1):
    #     # break
    #     with torch.no_grad():
    #         image = data[0].cuda()
    #         out = model(image)
    #         fea = out[1]
    #         feas.append(fea)
    #
    # feas = torch.cat(feas)
    # # print('duke_train_feas', feas.size())
    # return feas.cpu().numpy()
