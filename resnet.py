# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""ResNet."""
import math
import re
from t_resnet import ResNet as t_ResNet
import pickle
import torch
import torch.nn as tnn

import numpy as np
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore import save_checkpoint, context, load_checkpoint, load_param_into_net
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from config.resnet_config import config
import mindspore


# import pydevd_pycharm
# pydevd_pycharm.settrace('120.26.167.153', port=8023, stdoutToServer=True, stderrToServer=True)

def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    if config.net_name == "resnet152":
        stddev = (scale ** 0.5)
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            neg_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + neg_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Unsupported mode {}, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_channel, out_channel, stride=1, ):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))

    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1, ):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))

    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1, ):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))

    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _bn(channel, ):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9, gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel, ):
    weight_shape = (out_channel, in_channel)
    weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))

    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class Bottleneck(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        inplanes (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        use_se (bool): Enable SE-ResNet50 net. Default: False.
        se_block(bool): Use se block in SE-ResNet50 net. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> Bottleneck(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, has_bias=False, pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()  # kuhn edited  there is no inplace parameter in ReLU
        self.down_sample = downsample

        # if stride != 1 or in_channel != out_channel:
        #     self.down_sample = True
        # self.down_sample_layer = None
        #
        # if self.down_sample:
        #     self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
        #                                                          ), _bn(out_channel)])

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layers (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
        use_se (bool): Enable SE-ResNet50 net. Default: False.
        se_block(bool): Use se block in SE-ResNet50 net in layer 3 and layer 4. Default: False.
        res_base (bool): Enable parameter setting of resnet18. Default: False.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(Bottleneck,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self, block, layers, num_classes, train=True):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.istrain = train
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, has_bias=False, pad_mode="pad")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")  # kuhn edited. THrought experience, the torch version's padding mode seems to be "valid". I am not sure.
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool2d = nn.AvgPool2d((16, 8))  # kuhn edited. might be wrong.

        self.num_features = 512
        self.feat = nn.Dense(512 * block.expansion, self.num_features, )  # kuhn edited. I assueme that the weights would be replaced by ckpts,so I didn't set any speical init weights.
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.classifier = nn.Dense(self.num_features, num_classes, )

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            se_block(bool): Use se block in SE-ResNet50 net. Default: False.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(Bottleneck, 3, 128, 256, 2)
        """
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion)])
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.SequentialCell(*layers)

        # resnet_block = block(in_channel, out_channel, stride=stride, )
        # layers.append(resnet_block)
        # for _ in range(1, layer_num):
        #     resnet_block = block(out_channel, out_channel, stride=1, )
        #     layers.append(resnet_block)
        # return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print("######################################## 1")
        print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print("######################################## 2")
        print(x.shape)
        x = self.avgpool2d(x)  # kuhn edited. Might cause error.
        # x = self.avgpool2d(x, x.shape[2:])
        x = x.view(x.shape[0], -1)

        x = self.feat(x)
        fea = self.feat_bn(x)
        fea_norm = P.L2Normalize()(fea)

        x = P.ReLU()(fea)
        x = self.classifier(x)
        return x, fea_norm, fea


class tBottleneck(tnn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(tBottleneck, self).__init__()
        self.conv1 = tnn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = tnn.BatchNorm2d(planes)
        self.conv2 = tnn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = tnn.BatchNorm2d(planes)
        self.conv3 = tnn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = tnn.BatchNorm2d(planes * 4)
        self.relu = tnn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def weights_converter():
    par_dict = torch.load("checkpoint/resnet50_duke2market_epoch00100.pth", map_location='cpu')
    params_list = []
    f1 = open("ms_model_param_name.txt", "r")
    f2 = open("torch_model_param_name.txt", "r")
    lines_f1 = f1.readlines()
    lines_f2 = f2.readlines()
    for i in range(len(lines_f1)):
        param_dict = {}
        param_dict["name"] = lines_f1[i].strip()
        param_dict['data'] = Tensor(par_dict[lines_f2[i].strip()].numpy())
        params_list.append(param_dict)

    save_checkpoint(params_list, 'ms_resnet50.ckpt')
    f1.close()
    f2.close()


def print_ms_model_param_name(net=None, file_name=None):
    if net is None:
        net = ResNet(Bottleneck, [3, 4, 6, 3], config.class_num, train=False)
    if file_name is None:
        file_name = "ms_model_param_name.txt"
    with open(file_name, 'w') as f:
        for item in net.get_parameters():
            f.write(str(item.name) + '\n')
            # f.write(str(item.name) + ' ' + str(item.shape) + '\n')
        # f.write(str(net))


def load_ms_model(net=None, checkpoint=None):
    if net is None:
        net = ResNet(Bottleneck, [3, 4, 6, 3], config.class_num, train=False)
    if checkpoint is None:
        checkpoint = "ms_resnet50.ckpt"
    params_dict = load_checkpoint(checkpoint)
    load_param_into_net(net, params_dict)
    return net


def t_resnet50(pretrained=None, num_classes=1000, train=True):
    model = t_ResNet(tBottleneck, [3, 4, 6, 3], num_classes, train)
    weight = torch.load(pretrained, map_location='cpu')
    static = model.state_dict()

    base_param = []
    for name, param in weight.items():
        if name not in static:
            continue
        if isinstance(param, tnn.Parameter):
            param = param.data
        static[name].copy_(param)
        base_param.append(name)

    params = []
    params_dict = dict(model.named_parameters())
    for key, v in params_dict.items():
        if key in base_param:
            params += [{'params': v, 'lr_mult': 1}]
        else:
            # new parameter have larger learning rate
            params += [{'params': v, 'lr_mult': 10}]

    return model, params


def load_torch_model():
    net, _ = t_resnet50(pretrained="checkpoint/resnet50_duke2market_epoch00100.pth", num_classes=702, train=False)
    # net = ResNet(Bottleneck, [3, 4, 6, 3], config.class_num, train=False)
    # net.load_state_dict(torch.load("checkpoint/resnet50_duke2market_epoch00100.pth", map_location='cpu'))
    return net


def torchVSms():
    m_net = load_ms_model()
    t_net = load_torch_model()

    test_input = np.random.randn(6, 3, 256, 128).astype(np.float32)
    m_tensor_in = Tensor(test_input)
    t_tensor_in = torch.from_numpy(test_input)

    t_tensor_out, _, _ = t_net(t_tensor_in)
    m_tensor_out, _, _ = m_net(m_tensor_in)
    print(m_tensor_out.shape)
    print("########################################")
    print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy()))
    print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy(), atol=1e-4))
    print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy(), atol=1e-3))
    print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy(), atol=1e-2))
    print(np.allclose(m_tensor_out.asnumpy(), t_tensor_out.detach().numpy(), atol=1e-1))
    with open("ms_tensor_out.txt", "w") as f:
        f.write(str(m_tensor_out.asnumpy()))
    with open("torch_tensor_out.txt", "w") as f:
        f.write(str(t_tensor_out.detach().numpy()))


if __name__ == '__main__':
    torchVSms()
    # weights_converter()
    # print_ms_model_param_name()
    # process_weights_txt()
    # ##########nhuk#################################### MindSpore
    # net = ResNet(Bottleneck, [3, 4, 6, 3], config.class_num, train=False)
    # with open('ms_model.txt', 'w') as f:
    #     for item in net.get_parameters():
    #         f.write(str(item.name) + ' ' + str(item.shape) + '\n')
    #     # f.write(str(net))
    # ##########nhuk####################################
    # context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend',save_graphs=True, save_graphs_path='./graphs')
    # #########nhuk#################################### torch
    # par_dict = torch.load("checkpoint/resnet50_duke2market_epoch00100.pth", map_location='cpu')
    # with open("torch_model_param_name.txt", 'w') as f:
    #     for i in par_dict.keys():
    #         f.write(i + '\n')
    # #########nhuk####################################
    # #########nhuk####################################
    # par_dict = torch.load("checkpoint/resnet50_duke2market_epoch00100.pth", map_location='cpu')
    # with open("torch_model_param_name.txt", 'w') as f:
    #     for i in par_dict.keys():
    #         if not "num_batches_tracked" in i:
    #             f.write(i + '\n')
    # #########nhuk####################################

    # tnet = tBottleneck(inplanes=64, planes=64, stride=1, downsample=None)
    # print(tnet)
    # tnet.load_state_dict(par_dict)
    # mnet = Bottleneck(inplanes=64, planes=64, stride=1, downsample=None)

    # par_dict = torch.load("checkpoint/resnet50_duke2market_epoch00100.pth", map_location='cpu')
    # conv1_weight = par_dict['conv1.weighttorch']
    # print(conv1_weight.shape)

    # using re
    # bn_weight_pattern = re.compile(r'bn\d+\.weight')
    # bn_bias_pattern = re.compile(r'bn\d+\.bias')
    # bn_running_mean_pattern = re.compile(r'bn\d+\.running_mean')
    # bn_running_var_pattern = re.compile(r'bn\d+\.running_var')

    print("hello")
