import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, mask_nodata_value: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if mask_nodata_value >= 0:
        ignore_mask = torch.eq(target, mask_nodata_value)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = mask_nodata_value
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, mask_nodata_value: int = -100, smooth=1e-6):

    # Average of Dice coefficient for all batches, or for a single mask

    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if mask_nodata_value >= 0:
            roi_mask = torch.ne(t_i, mask_nodata_value)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + smooth) / (sets_sum + smooth)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, mask_nodata_value: int = -100, smooth=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], mask_nodata_value, smooth)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, mask_nodata_value: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, mask_nodata_value=mask_nodata_value)