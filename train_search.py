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
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from search_net import Raincleaner_search

parser = argparse.ArgumentParser(description="searching")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[10,20,25], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument('--alpha_lr', type=float, default=2e-3, help='lr for alpha')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for alpha')
parser.add_argument("--save_path", type=str, default="logs/PReNet_test", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/Rain12600",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="2,3,4,7", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('Loading dataset ...\n')
    dataset_train = Dataset_train(data_path=opt.data_path,patch_size=128)
    dataset_valid = Dataset_valid(data_path=opt.data_path,patch_size=128)
    loader_train = DataLoader(dataset=dataset_train, num_workers=16, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=16, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of valid_for_search samples: %d\n" % int(len(dataset_valid)))

    # Build model
    basemodel = Raincleaner_search(space_name='exss')

    # loss function
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = nn.DataParallel(basemodel).cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(basemodel.get_weights(), lr=opt.lr)
    alpha_optim = optim.Adam(basemodel.get_alphas(), lr=opt.alpha_lr, betas=(0.5, 0.999), weight_decay=opt.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)
    arch_scheduler = MultiStepLR(alpha_optim, milestones=opt.milestone, gamma=0.1)

    # start training
    step = 0
    genotypes = {}
    for epoch in range(opt.epochs):
        lr, arch_lr = scheduler.get_last_lr()[0], arch_scheduler.get_last_lr()[0]
        alpha_losses, weight_losses, valid_losses, train_psnrs, valid_psnrs = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        print('base learning rate: {:.6f} and arch learning rate: {:.6f}'.format(lr, arch_lr))

        model.train()

        ## epoch training start
        arch_iter = iter(loader_valid)
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(non_blocking = True), target_train.cuda(non_blocking = True)

            try:
                input_valid, target_valid = next(arch_iter)
            except:
                arch_iter = iter(loader_valid)
                input_valid, target_valid = next(arch_iter)

            if opt.use_gpu:
                input_valid, target_valid = input_valid.cuda(non_blocking = True), target_valid.cuda(non_blocking = True)

            alpha_optim.zero_grad()
            out_valid = model(input_valid)
            valid_pixel_metric = criterion(target_valid, out_valid)
            arch_loss = -valid_pixel_metric

            arch_loss.backward()
            alpha_optim.step()

            optimizer.zero_grad()
            out_train = model(input_train)
            N = out_train.size(0)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            loss.backward()
            optimizer.step()

            alpha_losses.update(arch_loss.item(),N)
            weight_losses.update(loss.item(),N)

            out_train = torch.clamp(out_train, 0., 1.)
            out_valid = torch.clamp(out_valid, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            psnr_valid = batch_PSNR(out_valid, target_valid, 1.)
            train_psnrs.update(psnr_train.item(),N)
            valid_psnrs.update(psnr_valid.item(),N)
            print("[epoch %d][%d/%d] alpha loss: %.4f, alpha_pixel_metric: %.4f, weight loss: %.4f, weight_pixel_metric: %.4f, TRAIN_PSNR: %.4f, VALID_PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), alpha_losses.avg, valid_pixel_metric, weight_losses.avg, pixel_metric, train_psnrs.avg, valid_psnrs.avg))

            step += 1
        scheduler.step()
        arch_scheduler.step()
 
        genotypes[epoch] = basemodel.genotype()
        print("The {:}/{:}-th Genotype = {:}".format(epoch,opt.epochs,genotypes[epoch]))
        basemodel.print_alphas()

    print("Architecture Search Finished")
    torch.cuda.empty_cache()



if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrain200H') != -1:
            print(opt.data_path.find('RainTrain200H'))
            prepare_data_aug_RainTrain200H(data_path=opt.data_path, patch_size=128, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
