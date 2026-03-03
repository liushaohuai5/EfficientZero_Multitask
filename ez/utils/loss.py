import torch
# import torchrl
import torch.nn as nn
import torch.nn.functional as F
from ez.utils.distribution import SquashedNormal, TruncatedNormal, ContDist
from ez.utils.format import atanh, fsq_conversion, log_std_transform, gaussian_logprob, squash
from ez.utils.format import symlog, symexp, DiscreteSupport
from torch.cuda.amp import autocast as autocast
import math


def cosine_similarity_loss(f1, f2):
    """Cosine Consistency loss function: similarity loss
    Parameters
    """
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)

def kl_loss(prediction, target):
    return -(torch.log_softmax(prediction, dim=-1) * target).sum(-1)

def symlog_loss(prediction, target):
    return 0.5 * (prediction.squeeze() - symlog(target)) ** 2

def Value_loss(preds, targets, config):
    v_num = config.train.v_num
    targets = targets.repeat(v_num, 1)
    iql_weight = config.train.IQL_weight
    if not config.train.use_IQL:
        iql_weight = 0.5
    if config.model.value_support.type == 'symlog':
        loss_func = symlog_loss
        reformed_values = symexp(preds).squeeze()
        target_supports = targets
    elif config.model.value_support.type == 'support':
        loss_func = kl_loss
        reformed_values = DiscreteSupport.vector_to_scalar(preds, **config.model.value_support).squeeze()
        target_supports = DiscreteSupport.scalar_to_vector(targets, **config.model.value_support)
    else:
        raise NotImplementedError

    value_error = reformed_values - targets
    value_sign = (value_error > 0).float().detach()
    value_weight = (1 - value_sign) * iql_weight + value_sign * (1 - iql_weight)
    value_loss = (value_weight * loss_func(preds, target_supports)).mean(0)
    return value_loss

def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)
