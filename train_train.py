import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.data import DataLoader
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from SSIM import SSIM
from genotypes_searched import architectures
from train_model import Raincleaner_train

parser = argparse.ArgumentParser(description="search_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[20,40,60], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--lr_min", type=float, default=1e-5, help="min learning rate")
parser.add_argument("--save_path", type=str, default="logs/PReNet_test", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/Rain12600",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="4", help='GPU id')
parser.add_argument("--arch", type=str, help='Arch')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path,patch_size=128)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    genotype = architectures[opt.arch]
    model = Raincleaner_train(genotype) 
    print_network(model)

    # loss function
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = nn.DataParallel(model).cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()


            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            out_train = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out_train = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            step += 1
        # save model
        if not os.path.isdir(opt.save_path):
            os.mkdir(opt.save_path)
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrain200H') != -1:
            print(opt.data_path.find('RainTrain200H'))
            prepare_data_aug_RainTrain200H(data_path=opt.data_path, patch_size=128, stride=80)
        elif opt.data_path.find('DerainDrop') != -1:
            prepare_data_aug_DerainDrop(data_path=opt.data_path, patch_size=128, stride=80)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
