import cv2 as cv

cv.namedWindow('img',cv.WINDOW_NORMAL)


def look_img_filename(filename):
    im = cv.imread(filename)
    cv.imshow('img', im)
    cv.waitKey(0)


def look_img(im):
    cv.imshow('img', im)
    cv.waitKey(0)
