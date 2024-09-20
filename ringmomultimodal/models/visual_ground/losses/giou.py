import torch


def GIoU_Loss(boxes1, boxes2, size):
    '''
    cal GIOU of two boxes or batch boxes
    '''

    # ===========cal IOU=============#
    # cal Intersection
    bs = boxes1.size(0)
    boxes1 = torch.cat([boxes1[:, :2] - (boxes1[:, 2:] / 2), boxes1[:, :2] + (boxes1[:, 2:] / 2)], dim=1)
    boxes1 = torch.clamp(boxes1, min=0, max=size)
    max_xy = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    min_xy = torch.max(boxes1[:, :2], boxes2[:, :2])

    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, 0] * inter[:, 1]
    boxes1Area = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))
    boxes2Area = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]))

    union_area = boxes1Area + boxes2Area - inter + 1e-7
    ious = inter / union_area
    print("ious:", ious)

    # ===========cal enclose area for GIOU=============#
    enclose_left_up = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_right_down = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose = torch.clamp((enclose_right_down - enclose_left_up), min=0)
    enclose_area = enclose[:, 0] * enclose[:, 1] + 1e-7
    # cal GIOU
    gious = ious - 1.0 * (enclose_area - union_area) / enclose_area
    print("gious:", gious)
    print((gious > 1))
    # GIOU Loss
    giou_loss = ((1 - gious).sum()) / bs
    print("giou_loss", giou_loss, giou_loss > 0)
    return giou_loss
