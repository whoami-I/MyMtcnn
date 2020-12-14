import tensorflow as tf
import numpy as np
import os
import cv2 as cv

from src.data_process.constants import *
from src.utils.general import iou

# root_path代表项目的根路径
root_path = os.path.abspath(os.path.dirname(__file__)).split('MyMtcnn')[0]
root_path = os.path.join(root_path, 'MyMtcnn')
print(root_path)
IMAGE_PATH = os.path.join(root_path, 'train')
IMAGE_INDEX_FILE_PATH = os.path.join(root_path, 'train', 'trainImageList.txt')

NEGATIVE_INDEX_FILE_PATH = os.path.join(root_path, pnet_negative_data_dir, index_file_name)
NEGATIVE_IMAGE_DIR = os.path.join(root_path, pnet_negative_data_dir, img_file_dir)
POSITIVE_INDEX_FILE_PATH = os.path.join(root_path, pnet_positive_data_dir, index_file_name)
POSITIVE_IMAGE_DIR = os.path.join(root_path, pnet_positive_data_dir, img_file_dir)
PART_INDEX_FILE_PATH = os.path.join(root_path, pnet_part_data_dir, index_file_name)
PART_IMAGE_DIR = os.path.join(root_path, pnet_part_data_dir, img_file_dir)

# check dir exist
if not os.path.isdir(PART_IMAGE_DIR):
    os.makedirs(PART_IMAGE_DIR)

if not os.path.isdir(NEGATIVE_IMAGE_DIR):
    os.makedirs(NEGATIVE_IMAGE_DIR)

if not os.path.isdir(POSITIVE_IMAGE_DIR):
    os.makedirs(POSITIVE_IMAGE_DIR)

fp_negative_image_list_file = open(NEGATIVE_INDEX_FILE_PATH, 'w')
fp_part_image_list_file = open(PART_INDEX_FILE_PATH, 'w')
fp_positive_image_list_file = open(POSITIVE_INDEX_FILE_PATH, 'w')
stdsize = 12

total_idx = 0
pos_idx = neg_idx = part_idx = 0
# cv.namedWindow('image')
with open(IMAGE_INDEX_FILE_PATH) as fp:
    for line in fp:
        # print(line)

        if total_idx > 80000: break
        if line is None or line.isspace(): continue

        filename, x0, x1, y0, y1, *landmark = line.split()
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)
        im = cv.imread(os.path.join(IMAGE_PATH, filename))

        # 仅保留文件名
        filename = filename.split('\\')[-1]

        ## for visualize image
        # cv.rectangle(im, (x0, y0), (x1, y1), (255, 0, 0), 4)
        # cv.imshow('image', im)
        # cv.waitKey(0)
        height, width, channel = im.shape
        box = np.array([x0, y0, x1, y1])
        box_width = x1 - x0
        box_height = y1 - y0
        """
            生成负例，写入图片数据和文件名即可，不用写入边框信息
        """
        for i in range(8):
            size = np.random.randint(stdsize, min(width, height) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            iou_value = iou(crop_box, box)

            if iou_value < 0.2:
                crop_im = im[ny:ny + size, nx:nx + size, :]
                resize_im = cv.resize(crop_im, (stdsize, stdsize), interpolation=cv.INTER_LINEAR)
                prefix, suffix = filename.split('.')
                dst_img_file_simple_name = prefix + '_' + str(neg_idx) + '.' + suffix
                dst_img_file_name = os.path.join(NEGATIVE_IMAGE_DIR, dst_img_file_simple_name)
                neg_idx += 1
                total_idx += 1
                cv.imwrite(dst_img_file_name, resize_im)
                fp_negative_image_list_file.write('%s\n' % dst_img_file_simple_name)

        """
        生成正例和部分例，需要写入图片数据，文件名和边框的偏移量
        """

        for i in range(15):

            size = np.random.randint(int(min(box_width, box_height) * 0.8), np.ceil(1.25 * max(box_width, box_height)))
            # delta here is the offset of box center
            delta_x = np.random.randint(-box_width * 0.2, box_width * 0.2)

            delta_y = np.random.randint(-box_height * 0.2, box_height * 0.2)
            print(size, delta_x, delta_y)
            nx = max(x0 + box_width / 2 + delta_x - size / 2, 0)

            ny = max(y0 + box_height / 2 + delta_y - size / 2, 0)
            if ny + size > height or nx + size > width: continue
            crop_box = np.array([nx, ny, nx + size, ny + size])

            # 偏移是相对于size的，因此在还原的时候，也要相对于size而言,
            # 并且其实位置相对也是对应的边
            offset_x0 = (x0 - nx) / float(size)
            offset_y0 = (y0 - ny) / float(size)
            offset_x1 = (x1 - nx - size) / float(size)
            offset_y1 = (y1 - ny - size) / float(size)
            cropped_im = im[int(ny):int(ny + size), int(nx):int(nx + size), :]
            resized_im = cv.resize(cropped_im, (stdsize, stdsize), interpolation=cv.INTER_LINEAR)

            dst_img_file_name = None
            dst_index_file_name = None
            iou_value = iou(crop_box, box)
            # print(iou_value)
            if iou_value > 0.65:
                # 正例
                prefix, suffix = filename.split('.')
                dst_img_file_simple_name = prefix + '_' + str(pos_idx) + '.' + suffix
                dst_img_file_name = os.path.join(POSITIVE_IMAGE_DIR, dst_img_file_simple_name)
                pos_idx += 1
                total_idx += 1
                dst_index_file_name = POSITIVE_INDEX_FILE_PATH
                cv.imwrite(dst_img_file_name, resized_im)
                fp_positive_image_list_file.write('%s' % dst_img_file_simple_name + ' %.4f %.4f %.4f %.4f\n' % (
                    offset_x0, offset_y0, offset_x1, offset_y1))
            elif iou_value > 0.35:
                # 部分
                prefix, suffix = filename.split('.')
                dst_img_file_simple_name = prefix + '_' + str(part_idx) + '.' + suffix
                dst_img_file_name = os.path.join(PART_IMAGE_DIR, dst_img_file_simple_name)
                part_idx += 1
                total_idx += 1
                dst_index_file_name = PART_INDEX_FILE_PATH
                cv.imwrite(dst_img_file_name, resized_im)
                fp_part_image_list_file.write('%s' % dst_img_file_simple_name + ' %.4f %.4f %.4f %.4f\n' % (
                    offset_x0, offset_y0, offset_x1, offset_y1))

fp_negative_image_list_file.close()
fp_part_image_list_file.close()
fp_positive_image_list_file.close()
print('generate ', total_idx, ' image')
