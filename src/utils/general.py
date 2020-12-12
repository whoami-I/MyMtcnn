def iou(box1, box2):
    """
    calculate iou between box1 and box2
    :param box1: [x0,y0,x1,y1]
    :param box2: [x0,y0,x1,y1]
    :return:
    """
    xx0 = max(box1[0], box2[0])
    yy0 = max(box1[1], box2[1])
    xx1 = min(box1[2], box2[2])
    yy1 = min(box1[3], box2[3])
    inter_width = max(xx1 - xx0 + 1, 0)
    inter_height = max(yy1 - yy0 + 1, 0)
    inter_area = inter_width * inter_height
    union_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) + (box2[2] - box2[0] + 1) * (
                box2[3] - box2[1] + 1) - inter_area
    return inter_area / union_area
