import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from models.FFA import FFA
import torchvision.transforms as tfs
import numpy as np
import os
from PIL import Image


if __name__ == '__main__':
    test_data_path = './data/'
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(0, 1):
        net = FFA(gps=3, blocks=19)
        net = net.to(device)
        net = torch.nn.DataParallel(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        # ckp = torch.load('./NHHaze_train_ffa_3_19.pk')
        ckp = torch.load('./its_train_ffa_3_19.pk')
        net.load_state_dict(ckp['model'])
        net.eval()

        with torch.no_grad():
            hazy_path = test_data_path
            hazy_list = os.listdir(hazy_path)
            data = [hazy_path + '/' + hazy_list[i] for i in range(len(hazy_list))]

            if not os.path.exists('./output/image_epoch{}'.format(epoch)):
                os.mkdir('./output/image_epoch{}'.format(epoch))
            output_dir = './output/image_epoch{}/'.format(epoch)

            for index in range(len(hazy_list)):
                hazy = data[index]
                hazy = Image.open(hazy)
                if hazy.mode != 'RGB':
                    hazy = hazy.convert('RGB')
                hazy1 = tfs.Compose([
                    tfs.ToTensor(),
                    tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
                ])(hazy)[None, ::]
                # hazy = torch.Tensor(hazy).unsqueeze(dim=0).permute(0, 3, 1, 2)
                hazy_for_A = F.interpolate(hazy1, size=(256, 256), mode='bilinear')
                _, clean, _, _, _, _ = net(hazy1, hazy_for_A, True)
                clean = torch.squeeze(clean.clamp(0, 1).cpu())
                vutils.save_image(clean, output_dir + '{}_img.png'.format(index))

