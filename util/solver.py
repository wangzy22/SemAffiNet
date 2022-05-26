# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import torch
import logging

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR


class LambdaStepLR(LambdaLR):

  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v


class PolyLR(LambdaStepLR):
  """DeepLab learning rate policy"""

  def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)


class SquaredLR(LambdaStepLR):
  """ Used for SGD Lars"""

  def __init__(self, optimizer, max_iter, last_step=-1):
    super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**2, last_step)


class SquaredLRWarmingUp(LambdaStepLR):
  """ Used for SGD Lars"""

  def __init__(self, optimizer, max_iter, warmingup_iter=500, last_step=-1):
    lambda_s = lambda s: (1 - s / (max_iter + 1))**2 if s > warmingup_iter else s / warmingup_iter * ((1 - s / (max_iter + 1))**2 - 1e-6) + 1e-6
    super(SquaredLRWarmingUp, self).__init__(optimizer, lambda_s, last_step)


class ExpLR(LambdaStepLR):

  def __init__(self, optimizer, step_size, gamma=0.9, last_step=-1):
    # (0.9 ** 21.854) = 0.1, (0.95 ** 44.8906) = 0.1
    # To get 0.1 every N using gamma 0.9, N * log(0.9)/log(0.1) = 0.04575749 N
    # To get 0.1 every N using gamma g, g ** N = 0.1 -> N * log(g) = log(0.1) -> g = np.exp(log(0.1) / N)
    super(ExpLR, self).__init__(optimizer, lambda s: gamma**(s / step_size), last_step)


class SGDLars(SGD):
  """Lars Optimizer (https://arxiv.org/pdf/1708.03888.pdf)"""

  def step(self, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
        and returns the loss.

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      weight_decay = group['weight_decay']
      momentum = group['momentum']
      dampening = group['dampening']
      nesterov = group['nesterov']

      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data
        # LARS
        w_norm = torch.norm(p.data)
        lamb = w_norm / (w_norm + torch.norm(d_p))
        d_p.mul_(lamb)
        if weight_decay != 0:
          d_p.add_(weight_decay, p.data)
        if momentum != 0:
          param_state = self.state[p]
          if 'momentum_buffer' not in param_state:
            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            buf.mul_(momentum).add_(d_p)
          else:
            buf = param_state['momentum_buffer']
            buf.mul_(momentum).add_(1 - dampening, d_p)
          if nesterov:
            d_p = d_p.add(momentum, buf)
          else:
            d_p = buf

        p.data.add_(-group['lr'], d_p)

    return loss


def initialize_optimizer(params, config, mode):
  if mode == 'b':
    config_optimizer = config.optimizer_b
    config_lr = config.lr_b
  elif mode == 'p':
    config_optimizer = config.optimizer_p
    config_lr = config.lr_p
  else:
    raise ValueError('Invalid mode!')
  assert config_optimizer in ['SGD', 'Adagrad', 'Adam', 'RMSProp', 'Rprop', 'SGDLars', 'AdamW']

  if config_optimizer == 'SGD':
    return SGD(
        params,
        lr=config_lr,
        momentum=config.sgd_momentum,
        dampening=config.sgd_dampening,
        weight_decay=1e-4)
  if config_optimizer == 'SGDLars':
    return SGDLars(
        params,
        lr=config_lr,
        momentum=config.sgd_momentum,
        dampening=config.sgd_dampening,
        weight_decay=config.weight_decay)
  elif config_optimizer == 'Adam':
    return Adam(
        params,
        lr=config_lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay)
  elif config_optimizer == 'AdamW':
    return AdamW(
        params,
        lr=config_lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay)
  else:
    logging.error('Optimizer type not supported')
    raise ValueError('Optimizer type not supported')


def initialize_scheduler(optimizer, config, last_step=-1):
  if config.scheduler == 'StepLR':
    return StepLR(
        optimizer, step_size=config.step_size, gamma=config.step_gamma, last_epoch=last_step)
  elif config.scheduler == 'PolyLR':
    return PolyLR(optimizer, max_iter=config.max_iter, power=config.poly_power, last_step=last_step)
  elif config.scheduler == 'SquaredLR':
    return SquaredLR(optimizer, max_iter=config.max_iter, last_step=last_step)
  elif config.scheduler == 'ExpLR':
    return ExpLR(
        optimizer, step_size=config.exp_step_size, gamma=config.exp_gamma, last_step=last_step)
  elif config.scheduler == 'SquaredLRWarmingUp':
    return SquaredLRWarmingUp(optimizer, max_iter=config.max_iter, last_step=last_step)
  else:
    logging.error('Scheduler not supported')
