#coding:utf-8

import cv2
import numpy
import sys

file_ = './1.jpg'

def main():
    im = cv2.imread(file_)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
    print im.shape, im_gray.shape

if __name__ == '__main__':
    main()
