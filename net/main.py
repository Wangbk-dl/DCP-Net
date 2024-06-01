import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models import *
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils

warnings.filterwarnings('ignore')
from option import opt, model_name, log_dir
from data_utils import *
from torchvision.models import vgg16
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

print('log_dir :', log_dir)
print('model_name:', model_name)

models_ = {
    'ffa': FFA(gps=opt.gps, blocks=opt.blocks),
}
loaders_ = {
    'its_train': ITS_train_loader,
    'its_test': ITS_test_loader,
    # 'ots_train':OTS_train_loader,
    # 'ots_test':OTS_test_loader
}
start_time = time.time()
T = opt.steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train_stage1(net, loader_train, loader_test, optim, criterion):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        print(max_psnr)
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_step:{start_step} start training ---')
    else:
        print('train from scratch *** ')
    for step in range(start_step + 1, opt.steps + 1):
        net.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y, t_gt = next(iter(loader_train))
        x = x.to(opt.device)
        y = y.to(opt.device)
        t_gt = t_gt.to(opt.device)
        # print(x.shape)
        _, out_J, out_T, T_var, out_A, out_I = net(x1=x)
        # print(T_var.mean().data)
        t_var = torch.exp(-T_var)
        weighted_T = torch.mul(out_T, t_var)
        weighted_T_gt = torch.mul(t_gt, t_var)
        uncertainty_loss = F.l1_loss(weighted_T, weighted_T_gt) + 2 * torch.mean(T_var)
        # print(0.008 * uncertainty_loss.data)
        # print(criterion[0](out_J, y))
        loss = criterion[0](out_J, y) + 0.1 * criterion[0](out_I, x) + 0.01 * uncertainty_loss
        if opt.perloss:
            # loss2 = criterion[1](out, y)
            # loss = loss + 0.04 * loss2
            loss = loss

        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        print(
            f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)
        # print('')

        # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
        # writer.add_scalar('data/loss',loss,step)

        if step % opt.eval_step == 0:
            torch.save({
                'model': net.state_dict()
            }, './checkpoints/{}.pk'.format(step))
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)

            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
            # 	writer.add_scalar('data/ssim',ssim_eval,step)
            # 	writer.add_scalar('data/psnr',psnr_eval,step)
            # 	writer.add_scalars('group',{
            # 		'ssim':ssim_eval,
            # 		'psnr':psnr_eval,
            # 		'loss':loss
            # 	},step)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict()
                }, opt.model_dir)
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


def train_stage2(net, loader_train, loader_test, optim, criterion):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_step:{start_step} start training ---')
        print(max_psnr)
    else:
        print('train from scratch *** ')
    for step in range(start_step + 1, opt.steps + 1):
        net.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y, t_gt = next(iter(loader_train))
        x = x.to(opt.device)
        y = y.to(opt.device)
        t_gt = t_gt.to(opt.device)
        # print(x.shape)
        _, out_J, out_T, T_var, out_A, out_I = net(x1=x, stage='stage2')
        b, c, h, w = T_var.shape
        var = T_var.view(b, c, -1)
        p_min = torch.min(var, dim=-1)
        p_min = p_min[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        s = T_var - p_min + 1
        out_T1 = torch.mul(out_T, s)
        label = torch.mul(t_gt, s)
        uncertainty_loss = criterion[0](out_T1, label)
        loss = criterion[0](out_J, y) + 0.1 * criterion[0](out_I, x) + 0.1 * uncertainty_loss
        if opt.perloss:
            # loss2 = criterion[1](out, y)
            # loss = loss + 0.04 * loss2
            loss = loss

        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        print(
            f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
        # writer.add_scalar('data/loss',loss,step)

        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)

            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            # with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
            # 	writer.add_scalar('data/ssim',ssim_eval,step)
            # 	writer.add_scalar('data/psnr',psnr_eval,step)
            # 	writer.add_scalars('group',{
            # 		'ssim':ssim_eval,
            # 		'psnr':psnr_eval,
            # 		'loss':loss
            # 	},step)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict()
                }, opt.model_dir)
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    # s=True
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        reshaped_inputs = F.interpolate(inputs, size=(256, 256), mode='bilinear')
        targets = targets.to(opt.device)
        _, pred, _, _, _, _ = net(inputs, reshaped_inputs, True)
        # # print(pred)
        # tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
        # vutils.save_image(targets.cpu(),'target.png')
        # vutils.save_image(pred.cpu(),'pred.png')
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    # if (psnr1>max_psnr or ssim1 > max_ssim) and s :
    #		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
    #		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
    #		s=False
    return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    mode = 'stage1'
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    print(opt.perloss)
    # if opt.perloss:
    # 		vgg_model = vgg16(pretrained=True).features[:16]
    # 		vgg_model = vgg_model.to(opt.device)
    # 		for param in vgg_model.parameters():
    # 			param.requires_grad = False
    # 		criterion.append(PerLoss(vgg_model).to(opt.device))
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    if mode == 'stage1':
        train_stage1(net, loader_train, loader_test, optimizer, criterion)
    if mode == 'stage2':
        train_stage2(net, loader_train, loader_test, optimizer, criterion)
