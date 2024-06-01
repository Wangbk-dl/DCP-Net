import numpy as np
import torch
import random
from PIL import Image
import einops
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def prepare_data(path):
    if os.path.exists(path):
        # clean_path = path + 'clean'
        haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        haze_imgs = [os.path.join(path, 'hazy', img) for img in haze_imgs_dir]
        clear_dir = os.path.join(path, 'clear')
        if os.path.exists(real_path):
            # clean_list = os.listdir(clean_path)
            real_list = os.listdir(real_path)
            hazy_list = os.listdir(hazy_path)
            real_list.sort(key=lambda x: x[:4])
            hazy_list.sort(key=lambda x: x[:4])
            data = [{'real': real_path + '/' + real_list[i], 'hazy': hazy_path + '/' + hazy_list[i]}
                    for i in range(len(real_list))]
        else:
            raise FileNotFoundError('File not exist')
    else:
        raise FileNotFoundError('File not exist')

    idx = 0
    # clean_training = torch.zeros((3, 128, 128, len(clean_list)))
    real_training = []
    hazy_training = []
    A_training = []
    # for index in range(50):
    for index in range(int(len(real_list))):
        # clean = data[index]['clean']
        real = data[index]['real']
        hazy = data[index]['hazy']

        # index1 = hazy.find('_')
        # index2 = hazy[index1 + 1:].find('_')
        # index3 = hazy[index1 + index2 + 2:].find('_')
        # A = float(hazy[index1 + index2 + 2:index1 + index2 + index3 + 2])
        # A_training.append(A)
        # clean = Image.open(clean)
        # clean = np.array(clean)
        # clean = torch.Tensor(clean).unsqueeze(dim=0)
        # clean = torch.nn.functional.interpolate(clean, size=(128, 128, 3), mode='bilinear')
        # clean = clean.permute(2, 0, 1)
        # clean = clean / 255.
        # clean_training[:, :, :, idx] = clean

        real = Image.open(real)
        if real.mode != 'RGB':
            real = real.convert("RGB")
        real = np.array(real)
        real = torch.Tensor(real).unsqueeze(dim=0).permute(0, 3, 1, 2)
        real = real / 255.
        real_training.append(real.squeeze(dim=0))

        hazy = Image.open(hazy)
        if hazy.mode != 'RGB':
            hazy = hazy.convert("RGB")
        hazy = np.array(hazy)
        hazy = torch.Tensor(hazy).unsqueeze(dim=0).permute(0, 3, 1, 2)
        hazy = hazy / 255.
        hazy_training.append(hazy.squeeze(dim=0))

        idx = idx + 1
        print(idx)
    # clean_training[clean_training > 1.0] = 1.0
    # clean_training[clean_training < 0.] = 0.

    # real_training[real_training > 1.0] = 1.0
    # real_training[real_training < 0.] = 0.
    # hazy_training[hazy_training > 1.0] = 1.0
    # hazy_training[hazy_training < 0.] = 0.
    print('The number of training data: %04d' % idx)
    # return clean_training, fake_training, real_training
    return real_training, hazy_training


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
    clean = clean.unsqueeze(dim=0)
    clean = F.interpolate(clean, size=(128, 128), mode='bilinear')
    padded_image = torch.nn.functional.pad(clean, [win_size[0] // 2, win_size[0] // 2,
                                                   win_size[1] // 2, win_size[1] // 2], value=1)
    image_win = F.unfold(padded_image, kernel_size=[win_size[0], win_size[1]], stride=[1, 1])
    image_win = einops.rearrange(image_win, 'b c w -> b w c')
    gt_DCP_index = torch.min(image_win, dim=2)[1]

    image = hazy.unsqueeze(dim=0)
    image = F.interpolate(image, size=(128, 128), mode='bilinear')
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
    path = './data/ITS/'
    clean, hazy = prepare_data(path)
    for i in range(len(clean)):
    # for i in range(50):

        DCP_ = compute_gt_Dark_channel(clean[i], hazy[i])
        DCP_save = np.array(torch.Tensor.cpu(DCP_ * 255.0))
        plt.imshow(DCP_save, cmap='Greys_r')
        DCP_save = Image.fromarray(DCP_save).convert('L')
        DCP_save.save("./data/RESIDE_OTS/gt_DCP1/{:3d}_gt_DCP.bmp".format(i + 1))
        print(i)

        # dehazed_res = computer_dehazed_res(hazy[i], DCP)
        # dehazed_res = dehazed_res.squeeze(dim=0)
        # dehazed_save = np.array(torch.Tensor.cpu(dehazed_res.permute(1, 2, 0)))
        # dehazed_save = np.clip(dehazed_save, 0., 1.) * 255.
        # plt.imshow(dehazed_save)
        # dehazed_save = Image.fromarray(np.uint8(dehazed_save))
        # dehazed_save.save("./simulated_res/{:3d}_gt_dehazed.jpg".format(i + 1))

        # DCP, A = compute_transmittance(hazy[i])
        # DCP_save = np.array(torch.Tensor.cpu(DCP * 255.0))
        # trans_save = 255 - 0.95 * DCP_save
        # DCP_save = Image.fromarray(DCP_save).convert('L')
        # DCP_save.save("./simulated_res/{:3d}_estimated_T.bmp".format(i + 1))
        # plt.imshow(DCP_save, cmap='Greys_r')

        # clean_T = compute_transmittance(clean[i]).squeeze(dim=0).squeeze(dim=0)
        # out_save = np.array(torch.Tensor.cpu(clean_T * 255.0))
        # plt.imshow(out_save, cmap='Greys_r')
        # out_save = Image.fromarray(out_save)
        # out_save = out_save.convert('L')
        # if not os.path.exists("./generated_result"):
        #     os.mkdir("./generated_result")
        # out_save.save("./generated_result/{:3d}_gt.bmp".format(i + 1))
        # print(i)
        # out_save.show()

        # hazy_T, A = compute_transmittance(hazy[i])
        # dehazed_res = (hazy[i] - (1 - hazy_T) * A) / hazy_T
        # dehazed_res = dehazed_res.squeeze(dim=0)
        # dehazed_save = np.array(torch.Tensor.cpu(dehazed_res.permute(1, 2, 0)))
        # dehazed_save = np.clip(dehazed_save, 0., 1.) * 255.
        # plt.imshow(dehazed_save)
        # dehazed_save = Image.fromarray(np.uint8(dehazed_save))
        # dehazed_save.save("./simulated_res/{:3d}_dehazed.jpg".format(i + 1))
        # hazy_save = np.array(torch.Tensor.cpu(hazy_T.squeeze(dim=0) * 255.0))
        # plt.imshow(hazy_save, cmap='Greys_r')
        # hazy_save = Image.fromarray(hazy_save)
        # hazy_save = hazy_save.convert('L')
        # hazy_save.save("./generated_result/{:3d}_hazy.bmp".format(i + 1))
