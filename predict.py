#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
import skimage
from skimage.measure import compare_psnr, compare_ssim


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()
    return args

def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)


def align_to_four(img):
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4

    img = img[0:a_row, 0:a_col]
    return img


def predict(image):
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()

    out = model(image)[-1]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :]*255.
    
    return out


if __name__ == '__main__':
    args = get_args()

    input_list = sorted(os.listdir(args.input_dir))
    gt_list = sorted(os.listdir(args.gt_dir))
    num = len(input_list)
    cumulative_psnr = 0
    cumulative_ssim = 0
    for i in range(num):
        print ('Processing image: %s'%(input_list[i]))
        img = cv2.imread(args.input_dir + input_list[i])
        if input_list[i].split('-')[0] == 'pie':
            gt_name = 'pie-norain-' + input_list[i].split('-')[-1]
        else:
            gt_name = 'norain-' + input_list[i].split('-')[-1]
        #gt_name = 'norain-' + input_list[i].split('-')[-1]
        #gt_name = input_list[i].split('_')[0] + '_clean.png'
        gt_name = input_list[i].split('x2')[0] + input_list[i].split('x2')[1]
        gt = cv2.imread(args.gt_dir + gt_name)
        save_img = img
        img = align_to_four(img)
        gt = align_to_four(gt)
        if img.shape[0] < gt.shape[0] or img.shape[1] < gt.shape[1]:
            gt = gt[0:img.shape[0], 0:img.shape[1]]
        elif img.shape[0] > gt.shape[0] or img.shape[1] > gt.shape[1]:
            img = img[0:gt.shape[0], 0:gt.shape[1]]
        result = img #predict(img)
        result = np.array(result, dtype = 'uint8')
        print("result.shape: {:}  and gt.shape: {:}".format(result.shape, gt.shape))
        cur_psnr = calc_psnr(result, gt)
        cur_ssim = calc_ssim(result, gt)
        print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
    print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))

