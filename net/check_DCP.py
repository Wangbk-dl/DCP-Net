import numpy as np
import torch
import random
from PIL import Image
import einops
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def prepare_data(path):
    if os.path.exists(path):
        # clean_path = path + 'clean'
        hazy_path = path + 'hazy/'
        clear_dir = path + 'clear/'
        haze_imgs_dir = os.listdir(hazy_path)

    else:
        raise FileNotFoundError('File not exist')

    haze_imgs = [os.path.join(path, 'hazy', img) for img in haze_imgs_dir]
    # for index in range(50):
    for index in range(len(haze_imgs)):
        # clean = data[index]['clean']
        haze = Image.open(haze_imgs[index])
        haze = np.array(haze)
        haze = torch.Tensor(haze).unsqueeze(dim=0).permute(0, 3, 1, 2)
        haze = haze / 255.

        img = haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]

        clear_name = id + '.jpg'
        clear = Image.open(os.path.join(clear_dir, clear_name))
        clear = np.array(clear)
        clear = torch.Tensor(clear).unsqueeze(dim=0).permute(0, 3, 1, 2)
        clear = clear / 255.

        trans_name = img.split('/')[-1][:-4]

        DCP_ = compute_gt_Dark_channel(clear, haze)

        DCP_save = np.array(torch.Tensor.cpu(DCP_ * 255.0))
        plt.imshow(DCP_save, cmap='Greys_r')
        DCP_save = Image.fromarray(DCP_save).convert('L')
        # trans_name = hazy_name[:10]

        DCP_save.save("./data/OTS/dcp/{}_DCP.bmp".format(trans_name))
        print(index)

    # clean_training[clean_training > 1.0] = 1.0
    # clean_training[clean_training < 0.] = 0.

    # real_training[real_training > 1.0] = 1.0
    # real_training[real_training < 0.] = 0.
    # hazy_training[hazy_training > 1.0] = 1.0
    # hazy_training[hazy_training < 0.] = 0.
    # print('The number of training data: %04d' % idx)
    # return clean_training, fake_training, real_training
    return


def compute_transmittance(image, win_size=None, omega=0.95):
    # image - input hazy image of size [Height,Width,Num_channels]
    # win_size - DCP window size, default is [15,15]
    # omega - DCP omega (residual haze) parameter, default is 0.95
    # outputs:
    # t - output coarse transmission map of size [Height,Width]
    # A - estimation of airlight vector of size [3,1]

    if win_size is None:
        win_size = [15, 15]
    image = image.unsqueeze(dim=0)
    b, c, h, w = image.shape
    # H = np.shape(image)[0]
    # W = np.shape(image)[1]
    # C = np.shape(image)[2]  # number of channels = 3
    patch_size = win_size[0] * win_size[1]
    padded_image = torch.nn.functional.pad(image, [win_size[0] // 2, win_size[0] // 2,
                                                   win_size[1] // 2, win_size[1] // 2], value=1)
    # padded_image = np.pad(image, (((win_size[0] - 1) / 2, (win_size[0] - 1) / 2),
    #                               ((win_size[1] - 1) / 2, (win_size[1] - 1) / 2), (0, 0)), 'edge')
    num_patches = h * w
    image_win = F.unfold(padded_image, kernel_size=[win_size[0], win_size[1]], stride=[1, 1])
    image_win = einops.rearrange(image_win, 'b c w -> b w c')
    # image_win = viewW(padded_image, (win_size[0], win_size[1], C)).reshape(num_patches, patch_size, C)
    initial_DCP = torch.min(image_win, dim=2)[0]
    DCP_sorted_idx = torch.argsort(initial_DCP, dim=-1, descending=True)
    brightest_pixel_coords = DCP_sorted_idx[:, :int(0.001 * num_patches)]
    reshaped_image = einops.rearrange(image, 'b c h w -> b (h w) c')
    A = torch.zeros((b, 3))
    for i in range(b):
        A[i][0] = torch.max(reshaped_image[i, :, :][brightest_pixel_coords[i]][0])
        A[i][1] = torch.max(reshaped_image[i, :, :][brightest_pixel_coords[i]][1])
        A[i][2] = torch.max(reshaped_image[i, :, :][brightest_pixel_coords[i]][2])
    A = A.unsqueeze(dim=-1).unsqueeze(dim=-1)
    image_win = einops.rearrange(image_win, 'b w (c k) -> b c k w', c=c)
    normalized_img = image_win / A  # b 3 k hw
    normalized_img = torch.min(normalized_img, dim=1)[0]
    DCP = torch.min(normalized_img, dim=1)[0]
    DCP = einops.rearrange(DCP, 'b (h w) -> b h w', h=h, w=w).squeeze(dim=0)

    # DCP = torch.min(torch.reshape(image_win / A.unsqueeze(dim=-1).unsqueeze(dim=-1), (b, num_patches, -1)), dim=-1)[0]
    # DCP = einops.rearrange(DCP, 'b (h w) -> b 1 h w', h=h, w=w).squeeze(dim=0)
    T = 1 - omega * DCP
    return T, A


def compute_DCP(image, win_size=None, omega=0.95):
    # image - input hazy image of size [Height,Width,Num_channels]
    # win_size - DCP window size, default is [15,15]
    # omega - DCP omega (residual haze) parameter, default is 0.95
    # outputs:
    # t - output coarse transmission map of size [Height,Width]
    # A - estimation of airlight vector of size [3,1]

    if win_size is None:
        win_size = [15, 15]
    image = image.unsqueeze(dim=0)
    b, c, h, w = image.shape
    # H = np.shape(image)[0]
    # W = np.shape(image)[1]
    # C = np.shape(image)[2]  # number of channels = 3
    patch_size = win_size[0] * win_size[1]
    padded_image = torch.nn.functional.pad(image, [win_size[0] // 2, win_size[0] // 2,
                                                   win_size[1] // 2, win_size[1] // 2], value=1)
    # padded_image = np.pad(image, (((win_size[0] - 1) / 2, (win_size[0] - 1) / 2),
    #                               ((win_size[1] - 1) / 2, (win_size[1] - 1) / 2), (0, 0)), 'edge')
    num_patches = h * w
    image_win = F.unfold(padded_image, kernel_size=[win_size[0], win_size[1]], stride=[1, 1])
    image_win = einops.rearrange(image_win, 'b c w -> b w c')
    # image_win = viewW(padded_image, (win_size[0], win_size[1], C)).reshape(num_patches, patch_size, C)
    initial_DCP = torch.min(image_win, dim=2)[0]
    DCP_ = einops.rearrange(initial_DCP, 'b (h w) -> b h w', h=h, w=w)
    return DCP_.squeeze(dim=0)


def compute_gt_Dark_channel(clean, hazy, win_size=None):
    if win_size is None:
        win_size = [15, 15]
    # clean = clean.unsqueeze(dim=0)
    # clean = F.interpolate(clean, size=(128, 128), mode='bilinear')
    padded_image = torch.nn.functional.pad(clean, [win_size[0] // 2, win_size[0] // 2,
                                                   win_size[1] // 2, win_size[1] // 2], value=1)
    image_win = F.unfold(padded_image, kernel_size=[win_size[0], win_size[1]], stride=[1, 1])
    image_win = einops.rearrange(image_win, 'b c w -> b w c')
    gt_DCP_index = torch.min(image_win, dim=2)[1]

    image = hazy
    # image = F.interpolate(image, size=(128, 128), mode='bilinear')
    b, c, h, w = image.shape
    # H = np.shape(image)[0]
    # W = np.shape(image)[1]
    # C = np.shape(image)[2]  # number of channels = 3
    patch_size = win_size[0] * win_size[1]
    padded_image = torch.nn.functional.pad(image, [win_size[0] // 2, win_size[0] // 2,
                                                   win_size[1] // 2, win_size[1] // 2], value=1)
    # padded_image = np.pad(image, (((win_size[0] - 1) / 2, (win_size[0] - 1) / 2),
    #                               ((win_size[1] - 1) / 2, (win_size[1] - 1) / 2), (0, 0)), 'edge')
    num_patches = h * w
    image_win = F.unfold(padded_image, kernel_size=[win_size[0], win_size[1]], stride=[1, 1])
    image_win = einops.rearrange(image_win, 'b c w -> b w c')
    # image_win = viewW(padded_image, (win_size[0], win_size[1], C)).reshape(num_patches, patch_size, C)
    initial_DCP = torch.min(image_win, dim=2)[0]
    DCP_sorted_idx = torch.argsort(initial_DCP, dim=-1, descending=True)
    brightest_pixel_coords = DCP_sorted_idx[:, :int(0.001 * num_patches)]
    reshaped_image = einops.rearrange(image, 'b c h w -> b (h w) c')
    A = torch.zeros((b, 3))
    for i in range(b):
        A[i][0] = torch.max(reshaped_image[i, :, :][brightest_pixel_coords[i]][0])
        A[i][1] = torch.max(reshaped_image[i, :, :][brightest_pixel_coords[i]][1])
        A[i][2] = torch.max(reshaped_image[i, :, :][brightest_pixel_coords[i]][2])
    A = A.unsqueeze(dim=-1).unsqueeze(dim=-1)
    image_win = einops.rearrange(image_win, 'b w (c k) -> b c k w', c=c)
    normalized_img = image_win / A
    normalized_img = einops.rearrange(normalized_img, 'b c k w -> b w (c k)')
    DCP = torch.zeros(b, h * w).cuda()
    for n in range(b):
        for m in range(normalized_img.shape[1]):
            DCP[n, m] = normalized_img[n, m, gt_DCP_index[n, m]]
    DCP = einops.rearrange(DCP, 'b (h w) -> b h w', h=h, w=w).squeeze(dim=0)
    return DCP


if __name__ == '__main__':
    path = './data/OTS/'
    # clean, hazy = prepare_data(path)
    # for i in range(len(clean)):
    # # for i in range(50):
    #
    #     DCP_ = compute_gt_Dark_channel(clean[i], hazy[i])
    #     DCP_save = np.array(torch.Tensor.cpu(DCP_ * 255.0))
    #     plt.imshow(DCP_save, cmap='Greys_r')
    #     DCP_save = Image.fromarray(DCP_save).convert('L')
    #     DCP_save.save("./data/RESIDE_OTS/gt_DCP1/{:3d}_gt_DCP.bmp".format(i + 1))
    #     print(i)
    prepare_data(path)
