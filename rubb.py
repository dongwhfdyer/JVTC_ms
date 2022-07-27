import torch
import mindspore
from mindspore.common.initializer import initializer, HeNormal
import numpy as np
import torch.nn as tnn
import mindspore.nn as mnn
from mindspore.ops import operations as P

# The following implements BatchNorm2d with MindSpore.
import torch
import mindspore.nn as nn
from mindspore import Tensor

net = nn.BatchNorm2d(num_features=2, momentum=0.8)
x = Tensor(np.array([[[[1, 2], [1, 2]], [[3, 4], [3, 4]]]]).astype(np.float32))
output = net(x)
print(output)
# Out:
# [[[[0.999995   1.99999]
#    [0.999995   1.99999]]
#
#   [[2.999985   3.99998]
#    [2.999985   3.99998]]]]


# The following implements BatchNorm2d with torch.
input_x = torch.tensor(np.array([[[[1, 2], [1, 2]], [[3, 4], [3, 4]]]]).astype(np.float32))
m = torch.nn.BatchNorm2d(2, momentum=0.2)
output = m(input_x)
print(output)
# Out:
# tensor([[[[-1.0000,  1.0000],
#           [-1.0000,  1.0000]],
#
#          [[-1.0000,  1.0000],
#           [-1.0000,  1.0000]]]], grad_fn=<NativeBatchNormBackward>)

