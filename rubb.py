import torch
import mindspore
from mindspore.common.initializer import initializer, HeNormal
import numpy as np
import torch.nn as tnn
import mindspore.nn as mnn
from mindspore.ops import operations as P

# ##############nhuk################################### test Max Pool2d
# # pool of square window of size=3, stride=2
# tm = tnn.MaxPool2d(3, stride=2, padding=1)
# tinput = torch.randn(20, 17, 50, 31)
# toutput = tm(tinput)
#
# # mm = mnn.MaxPool2d(kernel_size=3, stride=2,pad_mode="same") # same is not same
# mm = mnn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
# minput = P.StandardNormal()((20, 17, 50, 31))
# moutput = mm(minput)
# print(toutput.shape)
# print("########################################")
# print(moutput.shape)
# ###############nhuk#################################
#
##########nhuk#################################### test kaiming_normal_
tensor_test = torch.randn(17, 50, 31)
tensor1 = initializer(HeNormal(), tensor_test, mindspore.float32)
tensor2 = initializer('he_normal', [1, 4, 3], mindspore.float32)
print(tensor1)
print("########################################")
print(tensor2)
##########nhuk####################################
