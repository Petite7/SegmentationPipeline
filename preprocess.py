import torch
import torch.nn as nn
import albumentations as alb
import numpy as np
import pandas
from tqdm import tqdm
import csv
import cv2
import os

# Preprocess Hyper Parameters ########
IMAGE_PATH = r'Medical/train'
MASK_PATH = r'Medical/train_mask'
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
# ####################################


def guassian_blur(file, k=(9, 9), sigmax=0, sigmay=0):
    _img = cv2.GaussianBlur(cv2.imread(file), ksize=k, sigmaX=sigmax, sigmaY=sigmay)


def clahe_sharpen(file, new_file=None, cliplimit=2.0, titlegridsize=(8, 8)):
    image = cv2.imread(file, 0)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=titlegridsize)
    if new_file is not None:
        cv2.imwrite(new_file, clahe.apply(image))
    else:
        cv2.imwrite(file, clahe.apply(image))


def save_original_wh(image_path, file=r'original_wh'):
    path = os.listdir(image_path)
    _w, _h = 0, 0
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, image in enumerate(tqdm(path)):
            im = cv2.imread(os.path.join(image_path, image))
            w, h = im.shape[1::-1]
            writer.writerow([image, w, h])
            _w = _w if _w > w else w
            _h = _h if _h > h else h
    return _w, _h


def pad_to_shape(img_path, msk_path=None, new_image='', new_mask=None, width=512, height=512, file=r"original_image_wh.csv"):
    _w, _h = save_original_wh(img_path, file)
    assert _w <= width and _h <= height, f"Image Width and Height bigger than settings, got w = {_w}, h = {_h}"
    if not os.path.exists(new_image) and new_image != '' and new_mask != '':
        os.makedirs(new_image)
        if msk_path is not None:
            os.makedirs(new_mask)
    images = os.listdir(img_path)
    for idx, image in enumerate(tqdm(images)):
        aug_image = cv2.imread(os.path.join(img_path, image))
        w, h = aug_image.shape[1::-1]
        aug_image = cv2.copyMakeBorder(aug_image, top=height - h, bottom=0, left=0, right=width - w,
                                       borderType=cv2.BORDER_CONSTANT, value=0)
        cv2.imwrite(os.path.join(new_image, image), aug_image)
        if msk_path is not None:
            aug_mask = cv2.imread(os.path.join(msk_path, image))
            w, h = aug_mask.shape[1::-1]
            aug_mask = cv2.copyMakeBorder(aug_mask, top=height - h, bottom=0, left=0, right=width - w,
                                          borderType=cv2.BORDER_CONSTANT, value=0)
            cv2.imwrite(os.path.join(new_mask, image), aug_mask)


def crop_to_shape(img_path, new_path=None, file_path=r"original_wh.csv"):
    if (new_path is not None) and (not os.path.exists(new_path)):
        os.makedirs(new_path)
    with open(file_path, 'r') as f:
        for idx, line in enumerate(tqdm(f)):
            image_name, w, h = line.split(',')
            w, h = int(w), int(h)
            image = cv2.imread(os.path.join(img_path, image_name))
            _w, _h = image.shape[1::-1]
            crop_image = image[(_h-h):_h, 0:w]
            cv2.imwrite(os.path.join(new_path, image_name), crop_image)


if __name__ == '__main__':
    imgs = os.listdir(IMAGE_PATH)
    pad_to_shape(IMAGE_PATH, MASK_PATH, r'PMedical/train', r'PMedical/train_mask')
    for i, img in enumerate(tqdm(imgs)):
        clahe_sharpen(os.path.join(r'PMedical/train', img))
