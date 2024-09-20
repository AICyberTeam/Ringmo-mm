import torch


def reg_loss(output, target):
    sm_l1_loss = torch.nn.SmoothL1Loss(reduction='mean')

    loss_x1 = sm_l1_loss(output[:, 0], target[:, 0])
    loss_x2 = sm_l1_loss(output[:, 1], target[:, 1])
    loss_y1 = sm_l1_loss(output[:, 2], target[:, 2])
    loss_y2 = sm_l1_loss(output[:, 3], target[:, 3])

    return (loss_x1 + loss_x2 + loss_y1 + loss_y2)
