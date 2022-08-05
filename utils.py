import pdb

import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P


def get_info(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        # self.img_list = [os.path.join(dataset_dir, i.split()[0]) for i in lines]
        labels = [int(i.split()[1]) for i in lines]
        cam_ids = [int(i.split()[2]) for i in lines]
        frames = [int(i.split()[3]) for i in lines]

    return labels, cam_ids, frames


# def cluster(dist, rho=1.6e-3):
#     tri_mat = np.triu(dist, 1)
#     tri_mat = tri_mat[np.nonzero(tri_mat)]
#     tri_mat = np.sort(tri_mat, axis=None)
#     top_num = np.round(rho * tri_mat.size).astype(int)
#     eps = tri_mat[:top_num].mean()  # *2
#     # print('eps in cluster: {:.3f}'.format(eps))
#     cluster = DBSCAN(eps=eps, min_samples=1, metric='precomputed', n_jobs=8)
#     labels = cluster.fit_predict(dist)
#
#     return labels


def concat_using_numpy(ms_tensor):
    temp = np.concatenate([ms_tensor[i].asnumpy() for i in range(len(ms_tensor))], axis=0)
    # temp = np.concatenate([ms_tensor[i].asnumpy() for i in range(ms_tensor.size())], axis=0)
    ms_tensor_concat = Tensor(temp)
    return ms_tensor_concat


def extract_fea_camtrans(model, loader):
    feas = []
    K = 6
    batch_size = 1
    for data in loader.create_dict_iterator():
        columns_names_list = ['images' + str(i) for i in range(K)]

        # for ind in columns_names_list:
        #     data[ind] = P.ExpandDims()(data[ind], 0)
        # waiting_concat_ = (data[key] for key in columns_names_list)

        # cconcat_ = P.Concat()

        # numpy_data = [data[key].asnumpy() for key in columns_names_list]
        # concat_data = np.concatenate(numpy_data, axis=0)
        # concat_images = Tensor(concat_data)

        concat_images = P.Concat()((data['images0'], data['images1'], data['images2'], data['images3'], data['images4'], data['images5']))

        print("concat_images", concat_images.shape)
        out = model(concat_images)
        fea = out[2]
        print("fea", fea.shape)
        fea = fea.reshape(batch_size, K, -1)
        print("fea", fea.shape)
        fea = fea.mean(axis=1)
        print("fea", fea.shape)
        fea = P.L2Normalize()(fea)
        print("fea", fea.shape)
        feas.append(fea)

    feas = P.Concat()(feas)
    # feas = concat_using_numpy(feas)  # kuhn edted
    print("feas", feas.shape)

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

    feas = P.Concat(feas)
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
