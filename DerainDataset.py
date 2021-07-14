import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *



def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data_Rain12600(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rainy_image')
    target_path = os.path.join(data_path, 'ground_truth')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(14):
            input_file = "%d_%d.jpg" % (i+1, j+1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)
                train_num += 1

    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)

def prepare_data_DerainDrop(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'data')
    target_path = os.path.join(data_path, 'gt')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    gt_files = os.listdir(target_path)
    #for i in range(1800):
    for target_file in gt_files:
        #target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            #input_file = "rain-%d.png" % (i + 1)
            #input_file = target_file.split('_')[0] + '_rain.png'
            input_file = 'rain-'+target_file.split('-')[-1]
            if not os.path.exists(os.path.join(input_path,input_file)):
                input_file = target_file.split('_')[0] + '_rain.png'

            if os.path.exists(os.path.join(input_path,input_file)):

                input_img = cv2.imread(os.path.join(input_path,input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1
            else:
                print('Error, file does not exist')

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_RainTrainH(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_target.h5')
    save_input_path = os.path.join(data_path, str(patch_size)+'_train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "rain-%d.png" % (i + 1)

            if os.path.exists(os.path.join(input_path,input_file)): 

                input_img = cv2.imread(os.path.join(input_path,input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)

def prepare_data_rain_dataset(data_path, patch_size, stride):
    # train
    print('process training data')
    input1_path = os.path.join(data_path, 'rainstreak')
    input2_path = os.path.join(data_path, 'raindrop')
    input3_path = os.path.join(data_path, 'rainstreak_raindrop')
    target_path = os.path.join(data_path, 'gt')

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_target.h5')
    save_input_path = os.path.join(data_path, str(patch_size)+'_train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0

    rs_files = os.listdir(input1_path)
    rd_files = os.listdir(input2_path)
    rsd_files = os.listdir(input3_path)
    source_files = rs_files + rd_files + rsd_files
    random.shuffle(source_files)
    for im_file in source_files:
        if len(im_file.split('-')) == 2 and im_file.split('-')[0] == 'rain':
            input_file = os.path.join(input1_path, im_file)
        elif len(im_file.split('-')) == 2 and im_file.split('-')[0] == 'rd':
            input_file = os.path.join(input2_path, im_file)
        elif len(im_file.split('-')) == 3:
            input_file = os.path.join(input3_path, im_file)

        if os.path.exists(input_file):
            input_img = cv2.imread(input_file)
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])        

            target_file = 'norain-' + im_file.split('-')[-1]
            if os.path.exists(os.path.join(target_path,target_file)):
                target = cv2.imread(os.path.join(target_path,target_file))
                b, g, r = cv2.split(target)
                target = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)




def prepare_data_aug_RainTrainH(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path,'rain')
    target_path = os.path.join(data_path,'norain')

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_aug_target.h5')
    save_input_path = os.path.join(data_path, str(patch_size)+'_train_aug_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    target_files = os.listdir(target_path)
    random.shuffle(target_files)
    for target_file in target_files:
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            target_f = cv2.flip(target, 1) 
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            bf, gf, rf = cv2.split(target_f)
            target_f   = cv2.merge([rf, gf, bf])

            input_file = 'rain-'+target_file.split('-')[-1]

            if os.path.exists(os.path.join(input_path,input_file)):

                input_img = cv2.imread(os.path.join(input_path,input_file))
                input_img_f = cv2.flip(input_img, 1)
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])
                bf, gf, rf = cv2.split(input_img_f)
                input_img_f = cv2.merge([rf, gf, bf])

                H, W, C = input_img.shape

                size = 128
                for i in range(15):  # 15 patches
                    x1 = np.random.randint(W-size) # patch_size 128
                    y1 = np.random.randint(H-size)
                    x2 = x1 + size
                    y2 = y1 + size

                    crop_input  = input_img[y1:y2, x1:x2]
                    crop_target = target[y1:y2, x1:x2]
                    crop_input_flip = input_img_f[y1:y2, x1:x2]
                    crop_target_flip = target_f[y1:y2, x1:x2]
                    input_img_1 = np.float32(normalize(crop_input))
                    target_img = np.float32(normalize(crop_target))
                    input_img_2 = np.float32(normalize(crop_input_flip))
                    target_img_2 = np.float32(normalize(crop_target_flip))
                    input_data_1 = input_img_1.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_1)
                    target_data_1 = target_img.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_1)
                    train_num = train_num + 1
                    input_data_2 = input_img_2.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_2)
                    target_data_2 = target_img_2.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_2)
                    train_num = train_num + 1
                    if input_data_1.shape[1] <=1 or input_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', input_path,input_file)



    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)




def prepare_data_aug_RainTrain200H(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path,'rain/X2')
    target_path = os.path.join(data_path,'norain')

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_aug_target.h5')
    save_input_path = os.path.join(data_path, str(patch_size)+'_train_aug_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            target_f = cv2.flip(target, 1) 
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            bf, gf, rf = cv2.split(target_f)
            target_f   = cv2.merge([rf, gf, bf])

            input_file = "norain-%dx2.png" % (i + 1)

            if os.path.exists(os.path.join(input_path,input_file)):

                input_img = cv2.imread(os.path.join(input_path,input_file))
                input_img_f = cv2.flip(input_img, 1)
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])
                bf, gf, rf = cv2.split(input_img_f)
                input_img_f = cv2.merge([rf, gf, bf])

                H, W, C = input_img.shape

                size = 128
                for i in range(15):  # 15 patches
                    x1 = np.random.randint(W-size) # patch_size 128
                    y1 = np.random.randint(H-size)
                    x2 = x1 + size
                    y2 = y1 + size

                    crop_input  = input_img[y1:y2, x1:x2]
                    crop_target = target[y1:y2, x1:x2]
                    crop_input_flip = input_img_f[y1:y2, x1:x2]
                    crop_target_flip = target_f[y1:y2, x1:x2]
                    input_img_1 = np.float32(normalize(crop_input))
                    target_img = np.float32(normalize(crop_target))
                    input_img_2 = np.float32(normalize(crop_input_flip))
                    target_img_2 = np.float32(normalize(crop_target_flip))
                    input_data_1 = input_img_1.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_1)
                    target_data_1 = target_img.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_1)
                    train_num = train_num + 1
                    input_data_2 = input_img_2.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_2)
                    target_data_2 = target_img_2.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_2)
                    train_num = train_num + 1
                    if input_data_1.shape[1] <=1 or input_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', input_path,input_file)


    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_search_RainTrainH(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rain')
    target_path = os.path.join(data_path, 'norain')

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_aug_target.h5')
    save_input_path = os.path.join(data_path, str(patch_size)+'_train_aug_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    target_files = os.listdir(target_path)
    random.shuffle(target_files)
    for target_file in target_files:
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            target_f = cv2.flip(target, 1) 
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            bf, gf, rf = cv2.split(target_f)
            target_f   = cv2.merge([rf, gf, bf])

            input_file = 'rain-' + target_file.split('-')[-1]

            if os.path.exists(os.path.join(input_path,input_file)):

                input_img = cv2.imread(os.path.join(input_path,input_file))
                input_img_f = cv2.flip(input_img, 1)
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])
                bf, gf, rf = cv2.split(input_img_f)
                input_img_f = cv2.merge([rf, gf, bf])

                H, W, C = input_img.shape

                size = 128
                for i in range(15): 
                    x1 = np.random.randint(W-size)
                    y1 = np.random.randint(H-size)
                    x2 = x1 + size
                    y2 = y1 + size

                    crop_input  = input_img[y1:y2, x1:x2]
                    crop_target = target[y1:y2, x1:x2]
                    crop_input_flip = input_img_f[y1:y2, x1:x2]
                    crop_target_flip = target_f[y1:y2, x1:x2]
                    input_img_1 = np.float32(normalize(crop_input))
                    target_img = np.float32(normalize(crop_target))
                    input_img_2 = np.float32(normalize(crop_input_flip))
                    target_img_2 = np.float32(normalize(crop_target_flip))
                    input_data_1 = input_img_1.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_1)
                    target_data_1 = target_img.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_1)
                    train_num = train_num + 1
                    input_data_2 = input_img_2.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_2)
                    target_data_2 = target_img_2.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_2)
                    train_num = train_num + 1
                    if input_data_1.shape[1] <=1 or input_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', input_path,input_file)



    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


def prepare_data_aug_rain_dataset(data_path, patch_size, stride):
    # train
    print('process training data')
    input1_path = os.path.join(data_path, 'rainstreak')
    input2_path = os.path.join(data_path, 'raindrop')
    input3_path = os.path.join(data_path, 'rainstreak_raindrop')
    target_path = os.path.join(data_path, 'gt')

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_aug_target.h5')
    save_input_path = os.path.join(data_path, str(patch_size)+'_train_aug_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    rs_files = os.listdir(input1_path)
    rd_files = os.listdir(input2_path)
    rsd_files = os.listdir(input3_path)
    source_files = rs_files + rd_files + rsd_files
    random.shuffle(source_files)
    for im_file in source_files:
        if len(im_file.split('-')) == 2 and im_file.split('-')[0] == 'rain':
            input_file = os.path.join(input1_path, im_file)
        elif len(im_file.split('-')) == 2 and im_file.split('-')[0] == 'rd':
            input_file = os.path.join(input2_path, im_file)
        elif len(im_file.split('-')) == 3:
            input_file = os.path.join(input3_path, im_file)

     
        if os.path.exists(input_file):

            input_img = cv2.imread(input_file)
            input_img_f = cv2.flip(input_img, 1)
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])
            bf, gf, rf = cv2.split(input_img_f)
            input_img_f = cv2.merge([rf, gf, bf])

            target_file = 'norain-' + im_file.split('-')[-1]
            if os.path.exists(os.path.join(target_path,target_file)):
                target = cv2.imread(os.path.join(target_path,target_file))
                target_f = cv2.flip(target, 1) 
                b, g, r = cv2.split(target)
                target = cv2.merge([r, g, b])
                bf, gf, rf = cv2.split(target_f)
                target_f   = cv2.merge([rf, gf, bf])

                H, W, C = input_img.shape

                size = 128
                for i in range(10):  # 10 patches
                    x1 = np.random.randint(W-size) # patch_size 128
                    y1 = np.random.randint(H-size)
                    x2 = x1 + size
                    y2 = y1 + size

                    crop_input  = input_img[y1:y2, x1:x2]
                    crop_target = target[y1:y2, x1:x2]
                    crop_input_flip = input_img_f[y1:y2, x1:x2]
                    crop_target_flip = target_f[y1:y2, x1:x2]
                    input_img_1 = np.float32(normalize(crop_input))
                    target_img = np.float32(normalize(crop_target))
                    input_img_2 = np.float32(normalize(crop_input_flip))
                    target_img_2 = np.float32(normalize(crop_target_flip))
                    input_data_1 = input_img_1.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_1)
                    target_data_1 = target_img.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_1)
                    train_num = train_num + 1
                    input_data_2 = input_img_2.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_2)
                    target_data_2 = target_img_2.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_2)
                    train_num = train_num + 1
                    if input_data_1.shape[1] <=1 or input_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', input_path,input_file)


    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)




def prepare_data_aug_DerainDrop(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path,'data')
    target_path = os.path.join(data_path,'gt')

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_aug_target.h5')
    save_input_path = os.path.join(data_path, str(patch_size)+'_train_aug_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    gt_files = os.listdir(target_path)
    random.shuffle(gt_files)
    for target_file in gt_files:
        
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            target_f = cv2.flip(target, 1) 
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            bf, gf, rf = cv2.split(target_f)
            target_f   = cv2.merge([rf, gf, bf])
            input_file = target_file.split('_')[0]+'_rain.png'
            if os.path.exists(os.path.join(input_path,input_file)):

                input_img = cv2.imread(os.path.join(input_path,input_file))
                input_img_f = cv2.flip(input_img, 1)
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])
                bf, gf, rf = cv2.split(input_img_f)
                input_img_f = cv2.merge([rf, gf, bf])

                H, W, C = input_img.shape

                size = 128
                for i in range(15):  # 15 patches
                    x1 = np.random.randint(W-size) # patch_size 128
                    y1 = np.random.randint(H-size)
                    x2 = x1 + size
                    y2 = y1 + size

                    crop_input  = input_img[y1:y2, x1:x2]
                    crop_target = target[y1:y2, x1:x2]
                    crop_input_flip = input_img_f[y1:y2, x1:x2]
                    crop_target_flip = target_f[y1:y2, x1:x2]
                    input_img_1 = np.float32(normalize(crop_input))
                    target_img = np.float32(normalize(crop_target))
                    input_img_2 = np.float32(normalize(crop_input_flip))
                    target_img_2 = np.float32(normalize(crop_target_flip))
                    input_data_1 = input_img_1.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_1)
                    target_data_1 = target_img.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_1)
                    train_num = train_num + 1
                    input_data_2 = input_img_2.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_2)
                    target_data_2 = target_img_2.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_2)
                    train_num = train_num + 1
                    if input_data_1.shape[1] <=1 or input_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', input_path,input_file)

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)



def prepare_data_aug_Rain1200(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path,'rain')
    target_path = os.path.join(data_path,'gt')

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_aug_target.h5')
    save_input_path = os.path.join(data_path, str(patch_size)+'_train_aug_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    gt_files = os.listdir(target_path)
    random.shuffle(gt_files)
    for target_file in gt_files:
        
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            target_f = cv2.flip(target, 1) 
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            bf, gf, rf = cv2.split(target_f)
            target_f   = cv2.merge([rf, gf, bf])

            input_file = target_file
            if os.path.exists(os.path.join(input_path,input_file)):

                input_img = cv2.imread(os.path.join(input_path,input_file))
                input_img_f = cv2.flip(input_img, 1)
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])
                bf, gf, rf = cv2.split(input_img_f)
                input_img_f = cv2.merge([rf, gf, bf])

                H, W, C = input_img.shape

                size = 128
                for i in range(15):  # 15 patches
                    x1 = np.random.randint(W-size) # patch_size 128
                    y1 = np.random.randint(H-size)
                    x2 = x1 + size
                    y2 = y1 + size

                    crop_input  = input_img[y1:y2, x1:x2]
                    crop_target = target[y1:y2, x1:x2]
                    crop_input_flip = input_img_f[y1:y2, x1:x2]
                    crop_target_flip = target_f[y1:y2, x1:x2]
                    input_img_1 = np.float32(normalize(crop_input))
                    target_img = np.float32(normalize(crop_target))
                    input_img_2 = np.float32(normalize(crop_input_flip))
                    target_img_2 = np.float32(normalize(crop_target_flip))
                    input_data_1 = input_img_1.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_1)
                    target_data_1 = target_img.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_1)
                    train_num = train_num + 1
                    input_data_2 = input_img_2.transpose(2,0,1).copy()
                    input_h5f.create_dataset(str(train_num), data=input_data_2)
                    target_data_2 = target_img_2.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_2)
                    train_num = train_num + 1
                    if input_data_1.shape[1] <=1 or input_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', input_path,input_file)



    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)



def prepare_data_aug_RSRD(data_path, patch_size, stride):
    # train
    print('process training data')
    rd_path = os.path.join(data_path,'raindrop')
    rs_path = os.path.join(data_path,'rainstreak')
    rdrs_path = os.path.join(data_path,'rainstreak_raindrop')
    target_path = os.path.join(data_path,'gt')

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_aug_target.h5')
    save_rd_path = os.path.join(data_path, str(patch_size)+'_train_aug_rd.h5')
    save_rs_path = os.path.join(data_path, str(patch_size)+'_train_aug_rs.h5')
    save_rdrs_path = os.path.join(data_path, str(patch_size)+'_train_aug_rdrs.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    rd_h5f = h5py.File(save_rd_path, 'w')
    rs_h5f = h5py.File(save_rs_path, 'w')
    rdrs_h5f = h5py.File(save_rdrs_path, 'w')

    train_num = 0
    gt_files = os.listdir(target_path)
    random.shuffle(gt_files)
    for target_file in gt_files:
        
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            target_f = cv2.flip(target, 1) 
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            bf, gf, rf = cv2.split(target_f)
            target_f   = cv2.merge([rf, gf, bf])

            if len(target_file.split('-')) == 3:
                rd_file = 'pie-rd-'+target_file.split('-')[-1]
                rs_file = 'pie-rain-'+target_file.split('-')[-1]
                rdrs_file = 'pie-rd-rain-'+target_file.split('-')[-1]
            elif len(target_file.split('-')) == 2:
                rd_file = 'rd-'+target_file.split('-')[-1]
                rs_file = 'rain-'+target_file.split('-')[-1]
                rdrs_file = 'rd-rain-'+target_file.split('-')[-1]
            if os.path.exists(os.path.join(rd_path,rd_file)):

                rd_img = cv2.imread(os.path.join(rd_path,rd_file))
                rd_img_f = cv2.flip(rd_img, 1)
                b, g, r = cv2.split(rd_img)
                rd_img = cv2.merge([r, g, b])
                bf, gf, rf = cv2.split(rd_img_f)
                rd_img_f = cv2.merge([rf, gf, bf])


                rs_img = cv2.imread(os.path.join(rs_path,rs_file))
                rs_img_f = cv2.flip(rs_img, 1)
                bs, gs, rs = cv2.split(rs_img)
                rs_img = cv2.merge([rs, gs, bs])
                bsf, gsf, rsf = cv2.split(rs_img_f)
                rs_img_f = cv2.merge([rsf, gsf, bsf])


                rdrs_img = cv2.imread(os.path.join(rdrs_path,rdrs_file))
                rdrs_img_f = cv2.flip(rdrs_img, 1)
                bds, gds, rds = cv2.split(rdrs_img)
                rdrs_img = cv2.merge([rds, gds, bds])
                bdsf, gdsf, rdsf = cv2.split(rdrs_img_f)
                rdrs_img_f = cv2.merge([rdsf, gdsf, bdsf])


                H, W, C = rd_img.shape

                size = 128
                for i in range(15):  # 15 patches
                    x1 = np.random.randint(W-size) # patch_size 128
                    y1 = np.random.randint(H-size)
                    x2 = x1 + size
                    y2 = y1 + size

                    crop_rd  = rd_img[y1:y2, x1:x2]
                    crop_rs  = rs_img[y1:y2, x1:x2]
                    crop_rdrs  = rdrs_img[y1:y2, x1:x2]
                    crop_target = target[y1:y2, x1:x2]
                    crop_rd_flip = rd_img_f[y1:y2, x1:x2]
                    crop_rs_flip = rs_img_f[y1:y2, x1:x2]
                    crop_rdrs_flip = rdrs_img_f[y1:y2, x1:x2]
                    crop_target_flip = target_f[y1:y2, x1:x2]
                    rd_img_1 = np.float32(normalize(crop_rd))
                    rs_img_1 = np.float32(normalize(crop_rs))
                    rdrs_img_1 = np.float32(normalize(crop_rdrs))
                    target_img = np.float32(normalize(crop_target))
                    rd_img_2 = np.float32(normalize(crop_rd_flip))
                    rs_img_2 = np.float32(normalize(crop_rs_flip))
                    rdrs_img_2 = np.float32(normalize(crop_rdrs_flip))
                    target_img_2 = np.float32(normalize(crop_target_flip))
                    rd_data_1 = rd_img_1.transpose(2,0,1).copy()
                    rs_data_1 = rs_img_1.transpose(2,0,1).copy()
                    rdrs_data_1 = rdrs_img_1.transpose(2,0,1).copy()
                    rd_h5f.create_dataset(str(train_num), data=rd_data_1)
                    rs_h5f.create_dataset(str(train_num), data=rs_data_1)
                    rdrs_h5f.create_dataset(str(train_num), data=rdrs_data_1)
                    target_data_1 = target_img.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_1)
                    train_num = train_num + 1
                    rd_data_2 = rd_img_2.transpose(2,0,1).copy()
                    rs_data_2 = rs_img_2.transpose(2,0,1).copy()
                    rdrs_data_2 = rdrs_img_2.transpose(2,0,1).copy()
                    rd_h5f.create_dataset(str(train_num), data=rd_data_2)
                    rs_h5f.create_dataset(str(train_num), data=rs_data_2)
                    rdrs_h5f.create_dataset(str(train_num), data=rdrs_data_2)
                    target_data_2 = target_img_2.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_2)
                    train_num = train_num + 1
                    if rd_data_1.shape[1] <=1 or rd_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', rd_path,rd_file)
                    if rs_data_1.shape[1] <=1 or rs_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', rs_path,rs_file)
                    if rdrs_data_1.shape[1] <=1 or rdrs_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', rdrs_path,rdrs_file)

    target_h5f.close()
    rd_h5f.close()
    rs_h5f.close()
    rdrs_h5f.close()

    print('training set, # samples %d\n' % train_num)



def prepare_data_aug_RainDS(data_path, patch_size, stride):
    # train
    print('process training data')
    rd_path = os.path.join(data_path,'raindrop')
    rs_path = os.path.join(data_path,'rainstreak')
    rdrs_path = os.path.join(data_path,'rainstreak_raindrop')
    target_path = os.path.join(data_path,'gt')

    save_target_path = os.path.join(data_path, str(patch_size)+'_train_aug_target.h5')
    save_rd_path = os.path.join(data_path, str(patch_size)+'_train_aug_rd.h5')
    save_rs_path = os.path.join(data_path, str(patch_size)+'_train_aug_rs.h5')
    save_rdrs_path = os.path.join(data_path, str(patch_size)+'_train_aug_rdrs.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    rd_h5f = h5py.File(save_rd_path, 'w')
    rs_h5f = h5py.File(save_rs_path, 'w')
    rdrs_h5f = h5py.File(save_rdrs_path, 'w')

    train_num = 0
    gt_files = os.listdir(target_path)
    random.shuffle(gt_files)
    for target_file in gt_files:
        
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            target_f = cv2.flip(target, 1) 
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            bf, gf, rf = cv2.split(target_f)
            target_f   = cv2.merge([rf, gf, bf])

            rd_file = target_file
            rs_file = target_file
            rdrs_file = target_file
            if os.path.exists(os.path.join(rd_path,rd_file)):

                rd_img = cv2.imread(os.path.join(rd_path,rd_file))
                rd_img_f = cv2.flip(rd_img, 1)
                b, g, r = cv2.split(rd_img)
                rd_img = cv2.merge([r, g, b])
                bf, gf, rf = cv2.split(rd_img_f)
                rd_img_f = cv2.merge([rf, gf, bf])


                rs_img = cv2.imread(os.path.join(rs_path,rs_file))
                rs_img_f = cv2.flip(rs_img, 1)
                bs, gs, rs = cv2.split(rs_img)
                rs_img = cv2.merge([rs, gs, bs])
                bsf, gsf, rsf = cv2.split(rs_img_f)
                rs_img_f = cv2.merge([rsf, gsf, bsf])


                rdrs_img = cv2.imread(os.path.join(rdrs_path,rdrs_file))
                rdrs_img_f = cv2.flip(rdrs_img, 1)
                bds, gds, rds = cv2.split(rdrs_img)
                rdrs_img = cv2.merge([rds, gds, bds])
                bdsf, gdsf, rdsf = cv2.split(rdrs_img_f)
                rdrs_img_f = cv2.merge([rdsf, gdsf, bdsf])


                H, W, C = rd_img.shape


                size = 128
                for i in range(15):  # 15 patches
                    x1 = np.random.randint(W-size) # patch_size 128
                    y1 = np.random.randint(H-size)
                    x2 = x1 + size
                    y2 = y1 + size

                    crop_rd  = rd_img[y1:y2, x1:x2]
                    crop_rs  = rs_img[y1:y2, x1:x2]
                    crop_rdrs  = rdrs_img[y1:y2, x1:x2]
                    crop_target = target[y1:y2, x1:x2]
                    crop_rd_flip = rd_img_f[y1:y2, x1:x2]
                    crop_rs_flip = rs_img_f[y1:y2, x1:x2]
                    crop_rdrs_flip = rdrs_img_f[y1:y2, x1:x2]
                    crop_target_flip = target_f[y1:y2, x1:x2]
                    rd_img_1 = np.float32(normalize(crop_rd))
                    rs_img_1 = np.float32(normalize(crop_rs))
                    rdrs_img_1 = np.float32(normalize(crop_rdrs))
                    target_img = np.float32(normalize(crop_target))
                    rd_img_2 = np.float32(normalize(crop_rd_flip))
                    rs_img_2 = np.float32(normalize(crop_rs_flip))
                    rdrs_img_2 = np.float32(normalize(crop_rdrs_flip))
                    target_img_2 = np.float32(normalize(crop_target_flip))
                    rd_data_1 = rd_img_1.transpose(2,0,1).copy()
                    rs_data_1 = rs_img_1.transpose(2,0,1).copy()
                    rdrs_data_1 = rdrs_img_1.transpose(2,0,1).copy()
                    rd_h5f.create_dataset(str(train_num), data=rd_data_1)
                    rs_h5f.create_dataset(str(train_num), data=rs_data_1)
                    rdrs_h5f.create_dataset(str(train_num), data=rdrs_data_1)
                    target_data_1 = target_img.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_1)
                    train_num = train_num + 1
                    rd_data_2 = rd_img_2.transpose(2,0,1).copy()
                    rs_data_2 = rs_img_2.transpose(2,0,1).copy()
                    rdrs_data_2 = rdrs_img_2.transpose(2,0,1).copy()
                    rd_h5f.create_dataset(str(train_num), data=rd_data_2)
                    rs_h5f.create_dataset(str(train_num), data=rs_data_2)
                    rdrs_h5f.create_dataset(str(train_num), data=rdrs_data_2)
                    target_data_2 = target_img_2.transpose(2,0,1).copy()
                    target_h5f.create_dataset(str(train_num), data=target_data_2)
                    train_num = train_num + 1
                    if rd_data_1.shape[1] <=1 or rd_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', rd_path,rd_file)
                    if rs_data_1.shape[1] <=1 or rs_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', rs_path,rs_file)
                    if rdrs_data_1.shape[1] <=1 or rdrs_data_2.shape[1] <=1 or target_data_1.shape[1] <=1 or target_data_2.shape[1] <=1:
                        print('wrong', rdrs_path,rdrs_file)

    target_h5f.close()
    rd_h5f.close()
    rs_h5f.close()
    rdrs_h5f.close()

    print('training set, # samples %d\n' % train_num)











def prepare_data_RainTrainL(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "rain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)


class Dataset(udata.Dataset):
    def __init__(self, data_path='.', patch_size=128):
        super(Dataset, self).__init__()

        self.data_path = data_path
        self.patch_size = patch_size

        target_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_input.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)

class Dataset_train(udata.Dataset):
    def __init__(self, data_path='.', patch_size=128):
        super(Dataset_train, self).__init__()

        self.data_path = data_path
        self.patch_size = patch_size

        target_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_input.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')


        all_keys = list(target_h5f.keys())
        train_num = int(len(all_keys) * 0.7)
        self.keys = all_keys[:train_num]
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_input.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)

class Dataset_valid(udata.Dataset):
    def __init__(self, data_path='.', patch_size=128):
        super(Dataset_valid, self).__init__()

        self.data_path = data_path
        self.patch_size = patch_size

        target_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_input.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')


        all_keys = list(target_h5f.keys())
        train_num = int(len(all_keys) * 0.7)
        self.keys = all_keys[train_num:]
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_input.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)

class rdDataset(udata.Dataset):
    def __init__(self, data_path='.', patch_size=128):
        super(rdDataset, self).__init__()

        self.data_path = data_path
        self.patch_size = patch_size

        target_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_rd.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_rd.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(input), torch.Tensor(target), torch.Tensor(target), torch.Tensor(target)

class rsDataset(udata.Dataset):
    def __init__(self, data_path='.', patch_size=128):
        super(rsDataset, self).__init__()

        self.data_path = data_path
        self.patch_size = patch_size

        target_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_rs.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_rs.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input),torch.Tensor(target),torch.Tensor(target), torch.Tensor(input),torch.Tensor(target)

class rdsDataset(udata.Dataset):
    def __init__(self, data_path='.', patch_size=128):
        super(rdsDataset, self).__init__()

        self.data_path = data_path
        self.patch_size = patch_size

        target_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_target.h5')
        #input_path = os.path.join(self.data_path, str(patch_size)+'_train_aug_rdrs.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        #input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        #input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_target.h5')
        input_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_rdrs.h5')
        rd_input_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_rd.h5')
        rs_input_path = os.path.join(self.data_path, str(self.patch_size)+'_train_aug_rs.h5')
        #target_path = os.path.join(self.data_path, 'train_target.h5')
        #input_path = os.path.join(self.data_path, 'train_input.h5')


        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')
        rd_input_h5f = h5py.File(rd_input_path, 'r')
        rs_input_h5f = h5py.File(rs_input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])
        rd_input = np.array(rd_input_h5f[key])
        rs_input = np.array(rs_input_h5f[key])

        target_h5f.close()
        input_h5f.close()
        rd_input_h5f.close()
        rs_input_h5f.close()

        return torch.Tensor(input), torch.Tensor(rd_input), torch.Tensor(target), torch.Tensor(rs_input), torch.Tensor(target)


