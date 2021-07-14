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
from train_model import Raincleaner_wosha, Raincleaner_share

parser = argparse.ArgumentParser(description="search_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[15,30,45], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--lr_min", type=float, default=1e-5, help="min learning rate")
parser.add_argument("--save_path", type=str, default="logs", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="4", help='GPU id')
parser.add_argument("--rs_arch", type=str, help='Rain Streak Arch')
parser.add_argument("--rd_arch", type=str, help='Raindrop Arch')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    print('Loading dataset ...\n')
    dataset_rd = rdDataset(data_path=opt.data_path,patch_size=128)
    dataset_rs = rsDataset(data_path=opt.data_path,patch_size=128)
    dataset_train = rdsDataset(data_path=opt.data_path,patch_size=128)
    combine_dataset = dataset_rd+dataset_rs+dataset_train 
    loader_train = DataLoader(dataset=combine_dataset, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    rs_genotype = architectures[opt.rs_arch]
    rd_genotype = architectures[opt.rd_arch]
    model = Raincleaner_wosha(rs_genotype,rd_genotype)
    print_network(model)

    # loss function
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = nn.DataParallel(model).cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = CosineAnnealingLR(optimizer, opt.epochs, eta_min=opt.lr_min)

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

        for i, (input_rds, input_rd, target_train, input_rs, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()


            if opt.use_gpu:
                input_rds, input_rs, input_rd, target_train = input_rds.cuda(), input_rs.cuda(), input_rd.cuda(), target_train.cuda()

            rs_out_1, rs_out_2, rd_out_1, rd_out_2, rds_out = model(input_rds)
            pixel_metric = criterion(target_train, rds_out)
            loss_2 = -criterion(input_rd, rs_out_1)
            loss_3 = -criterion(target_train, rd_out_1)
            loss_4 = -criterion(target_train, rs_out_2)
            loss_5 = -criterion(input_rs, rd_out_2)
            loss = -4*pixel_metric + loss_2 + loss_3 + loss_4 + loss_5

            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out1,out2,out3,out4,out5 = model(input_rds)
            out_train = torch.clamp(out5, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, loss_2: %.4f, loss_3: %.4f, loss_4: %.4f, loss_5: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item(), pixel_metric.item(), psnr_train))

            step += 1

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainDS_syn') != -1:
            prepare_data_aug_RSRD(data_path=opt.data_path, patch_size=128, stride=80)
        elif opt.data_path.find('RainDS_real') != -1:
            prepare_data_aug_RainDS(data_path=opt.data_path, patch_size=128, stride=80)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
