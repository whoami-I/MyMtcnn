import sys
import numpy as np
import cv2
import os
import numpy.random as npr

stdsize = 12

anno_file = "label.txt"
# im_dir = "samples"
pos_save_dir = str(stdsize) + "/positive"
part_save_dir = str(stdsize) + "/part"
neg_save_dir = str(stdsize) + '/negative'
save_dir = "12"

def IoU(pr_box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    # print("随机锚框:",pr_box)

    box_area = (pr_box[2] - pr_box[0] + 1) * (pr_box[3] - pr_box[1] + 1)
    # print("随机面积box_area:",box_area)
    # print("(boxes[:, 2] - boxes[:, 0] + 1):",(boxes[:, 2] - boxes[:, 0] + 1))
    #XML真实区域 X2-X1 +1 = W   Y2-Y1 = H   W*H
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # print("真实面积area:",area)
    # print("probx[0]",pr_box[0])
    # boxes[:, 0]代表取boxes这个nx4矩阵所有行的第一列数据
    xx1 = np.maximum(pr_box[0], boxes[:, 0])
    # print("xx1",xx1)
    yy1 = np.maximum(pr_box[1], boxes[:, 1])
    # print("yy1",yy1)
    xx2 = np.minimum(pr_box[2], boxes[:, 2])
    # print("xx2",xx2)
    yy2 = np.minimum(pr_box[3], boxes[:, 3])
    # print("yy2",yy2)
    # compute the width and height of the bounding box
    # print("xx2-xx1",(xx2-xx1))
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    # inter_area = (xx1 - xx2 + 1) * (yy1 - yy2 + 1)
    # w = np.max(xx1,yy1)

    inter = w * h
    # print("inter",inter_area)
    ovr = inter / (box_area + area - inter)
    print("IOU:",ovr)
    return ovr

# 生成一系列文件夹用于存储三类样本
def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

mkr(save_dir)
mkr(pos_save_dir)
mkr(part_save_dir)
mkr(neg_save_dir)

# 生成一系列txt文档用于存储Positive，Negative，Part三类数据的信息
f1 = open(os.path.join(save_dir, 'pos_' + str(stdsize) + '.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_' + str(stdsize) + '.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_' + str(stdsize) + '.txt'), 'w')

# 读取label.txt
with open(anno_file, 'r') as f:
    annotations = f.readlines()
    del annotations[0]
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0

for annotation in annotations:
    # print("annotation",annotation)
    annotation = annotation.strip().split(' ')

    im_path = annotation[0]

    bbox = list(map(float,annotation[1:]))

    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2] += boxes[:, 0] - 1
    boxes[:, 3] += boxes[:, 1] - 1
    # print("boxes",boxes)
    img = cv2.imread(im_path)
    # print(img.shape)
    idx += 1
    if idx % 100 == 0:
        print(idx, "images done")

    height, width, channel = img.shape
    print(img.shape)
    neg_num = 0
    while neg_num < 50:
        # 生成随机数，对每张数据集中的图像进行切割，生成一系列小的图像
        size = npr.randint(stdsize, min(width, height) / 2)

        nx = npr.randint(0, width - size)

        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])
        # print(crop_box)
        # print("boxes",boxes)
        # 计算小的图像与标注产生的检测框之间的IoU
        Iou = IoU(crop_box, boxes)
        # print(Iou)
        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write(str(stdsize)+"/negative/%s"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1


    for box in boxes:
        print(box)
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # max(w, h) < 40：参数40表示忽略的最小的脸的大小
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 20 or x1 < 0 or y1 < 0:
            continue
        # 生成与gt有重叠的反面例子
        for i in range(5):
            size = npr.randint(stdsize, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))

            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)
            # cropped_im = img[ny: ny + size, nx: nx + size, :]
            cropped_im = img[ny1 : ny1 + size, nx1 : nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(str(stdsize)+"/negative/%s" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1



        # generate positive examples and part faces
        for i in range(20):

            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)

            delta_y = npr.randint(-h * 0.2, h * 0.2)
            print(size,delta_x,delta_y)
            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)

            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size

            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[int(ny1):int(ny2), int(nx1):int(nx2), :]

            resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)

            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write(str(stdsize)+"/positive/%s"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write(str(stdsize)+"/part/%s"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()
