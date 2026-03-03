import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

act_function = nn.functional.relu
activation = nn.ReLU


# Post Activated Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = nn.functional.relu(out)
        return out

# Residual block
class FCResidualBlock(nn.Module):
    def __init__(self, input_shape, hidden_shape):
        super(FCResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_shape, hidden_shape)
        self.bn1 = nn.BatchNorm1d(hidden_shape)
        self.linear2 = nn.Linear(hidden_shape, input_shape)
        self.bn2 = nn.BatchNorm1d(input_shape)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out = out + identity
        out = nn.functional.relu(out)
        return out


# def mlp(
#     input_size,
#     hidden_sizes,
#     output_size,
#     output_activation=nn.Identity,
#     activation=nn.ELU,
#     init_zero=False,
# ):
#     """
#     MLP layers
#     :param input_size:
#     :param hidden_sizes:
#     :param output_size:
#     :param output_activation:
#     :param activation:
#     :param init_zero:   bool, zero initialization for the last layer (including w and b).
#                         This can provide stable zero outputs in the beginning.
#     :return:
#     """
#     sizes = [input_size] + hidden_sizes + [output_size]
#     layers = []
#     for i in range(len(sizes) - 1):
#         if i < len(sizes) - 2:
#             act = activation
#             layers += [nn.Linear(sizes[i], sizes[i + 1]),
#                        nn.BatchNorm1d(sizes[i + 1]),
#                        act()]
#         else:
#             act = output_activation
#             layers += [nn.Linear(sizes[i], sizes[i + 1]),
#                        act()]
#
#     if init_zero:
#         layers[-2].weight.data.fill_(0)
#         layers[-2].bias.data.fill_(0)
#
#     return nn.Sequential(*layers)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-5, momentum=0.1):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.count = 1e3
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))

    def forward(self, x):
        if self.training:
            try:
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
                batch_count = x.shape[0]
            except:
                mean = x
                var = 0.0
                batch_count = 1
            self.running_mean, self.running_var, self.count = self.update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, mean, var, batch_count)
            global_mean = self.running_mean
            global_var = self.running_var
        else:
            global_mean = self.running_mean
            global_var = self.running_var
        x = (x - global_mean) / torch.sqrt(global_var + self.epsilon)
        return x

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


class SEM(nn.Module):
    def __init__(self, V, tau=1.0, **kwargs):
        super().__init__()
        self.V = V
        self.tau = tau

    def forward(self, x):
        shape = x.shape
        logits = x.view(*shape[:-1], -1, self.V)
        return nn.functional.softmax(logits / self.tau, -1).view(shape)



def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=activation,
    init_zero=False,
    use_bn=True,
    p_norm=False,
    dropout=0.,
    norm_type='layernorm'
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            if use_bn:
                layers += [
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                    nn.BatchNorm1d(sizes[i + 1]) if norm_type == 'batchnorm' else nn.LayerNorm(sizes[i + 1]),
                    activation()
                ]
            else:
                layers += [
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                    activation()
                ]
        else:
            if p_norm == True:
                layers += [PNorm()]
            if output_activation is not nn.Identity:
                layers += [
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                    nn.LayerNorm(sizes[i + 1]),
                    output_activation()
                ]
            else:
                layers += [
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.Identity()
                ]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)


# improved residual block from Pre-LN Transformer
# ref: http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf
class ImproveResidualBlock(nn.Module):
    def __init__(self, input_shape, hidden_shape):
        super(ImproveResidualBlock, self).__init__()
        self.ln1 = nn.LayerNorm(input_shape)
        self.linear1 = nn.Linear(input_shape, hidden_shape)
        self.linear2 = nn.Linear(hidden_shape, input_shape)

        # self.linear2 = nn.Linear(hidden_shape, hidden_shape)
        # if input_shape != hidden_shape:
        #     self.proj = nn.Linear(input_shape, hidden_shape)
        # else:
        #     self.proj = None

    def forward(self, x):
        identity = x
        out = self.ln1(x)
        out = self.linear1(out)
        out = act_function(out)
        out = self.linear2(out)

        # if self.proj is not None:
        #     out = out + self.proj(identity)
        # else:
        out = out + identity
        return out


# L2 norm layer on the next to last layer
class PNorm(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        assert len(x.shape) == 2
        return nn.functional.normalize(x, dim=1, eps=self.eps)