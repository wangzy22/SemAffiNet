import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_omni(preds, targets, mode='bce'):
    num_class = preds[0].shape[1]
    loss = []
    for lvl in range(len(preds)):
        pred = preds[lvl]
        target = targets[lvl][:, :num_class]
        if mode == 'bce':
            # weight = target / target.max(dim=1, keepdim=True)[0].clamp(min=1)
            l = F.binary_cross_entropy_with_logits(pred, target.clamp(0, 1).detach())
        elif mode == 'focal':
            l = focal_loss(pred, target)
        elif mode == 'dice':
            l = dice_loss(pred, target)
        else:
            raise NotImplementedError('Invalid mode!')
        loss.append(l)
    loss = torch.stack(loss).mean()
    return loss


def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()
        