import cv2 as cv

cv.namedWindow('img', cv.WINDOW_NORMAL)


def look_img_filename(filename):
    im = cv.imread(filename)
    cv.imshow('img', im)
    cv.waitKey(0)


def look_img(im):
    cv.imshow('img', im)
    cv.waitKey(0)


def look_img_filename_with_box(filename, box):
    im = cv.imread(filename)
    look_img_im_with_box(im, box)


def look_img_im_with_box(im, box):
    """
    :param im:
    :param box: [x0,y0,x1,y1]
    :return:
    """
    cv.rectangle(im, (box[0], box[1]), (box[2], box[3]), thickness=2, color=(255, 0, 0))
    cv.imshow('img', im)
    cv.waitKey(0)


def look_img_with_box_and_offset(im, box, offset):
    height, width, _ = im.shape
    cv.rectangle(im, (offset[0] * width + box[0], offset[1] * height + box[1]),
                 (offset[2] * width + box[2], offset[3] * height + box[3]), thickness=2, color=(255, 0, 0))
    cv.imshow('img', im)
    cv.waitKey(0)
