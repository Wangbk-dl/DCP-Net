#原文链接：https: // blog.csdn.net / weixin_40500230 / article / details / 93845890
import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np



# if var.size(1) > 1:
#     convert = var.new(1, 3, 1, 1)
#     convert[0, 0, 0, 0] = 65.738
#     convert[0, 1, 0, 0] = 129.057
#     convert[0, 2, 0, 0] = 25.064
#     var.mul_(convert).div_(256)
#     var = var.sum(dim=1, keepdim=True)
# vis_fea_map.draw_features(1, 1, abs(var).cpu().numpy(),"***.png")
def draw_features(width, height, x, savename):


    # for i in range(width * height):
    #     plt.subplot(height, width, i + 1)
    #     plt.axis('off')
    img = x[0, 0, :, :]
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8


    h = x.shape[2]
    w = x.shape[3]
    plt.figure(figsize=(w/100, h/100))
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.imshow(img)
    # plt.colorbar()
    plt.savefig(savename, pad_inches=0.0,)


    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    # cv2.imwrite(savename,img)



# class ft_net(nn.Module):
# 
#     def __init__(self):
#         super(ft_net, self).__init__()
#         model_ft = models.resnet50(pretrained=True)
#         self.model = model_ft
# 
#     def forward(self, x):
#         if True:  # draw features or not
#             x = self.model.conv1(x)
#             draw_features(8, 8, x.cpu().numpy(), "{}/f1_conv1.png".format(savepath))
# 
#             x = self.model.bn1(x)
#             draw_features(8, 8, x.cpu().numpy(), "{}/f2_bn1.png".format(savepath))
# 
#             x = self.model.relu(x)
#             draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))
# 
#             x = self.model.maxpool(x)
#             draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))
# 
#             x = self.model.layer1(x)
#             draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))
# 
#             x = self.model.layer2(x)
#             draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))
# 
#             x = self.model.layer3(x)
#             draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))
# 
#             x = self.model.layer4(x)
#             draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
#             draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))
# 
#             x = self.model.avgpool(x)
#             plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
#             plt.savefig("{}/f9_avgpool.png".format(savepath))
#             plt.clf()
#             plt.close()
# 
#             x = x.view(x.size(0), -1)
#             x = self.model.fc(x)
#             plt.plot(np.linspace(1, 1000, 1000), x.cpu().numpy()[0, :])
#             plt.savefig("{}/f10_fc.png".format(savepath))
#             plt.clf()
#             plt.close()
#         else:
#             x = self.model.conv1(x)
#             x = self.model.bn1(x)
#             x = self.model.relu(x)
#             x = self.model.maxpool(x)
#             x = self.model.layer1(x)
#             x = self.model.layer2(x)
#             x = self.model.layer3(x)
#             x = self.model.layer4(x)
#             x = self.model.avgpool(x)
#             x = x.view(x.size(0), -1)
#             x = self.model.fc(x)
# 
#         return x


# model = ft_net().cuda()
#
# # pretrained_dict = resnet50.state_dict()
# # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # model_dict.update(pretrained_dict)
# # net.load_state_dict(model_dict)
# model.eval()
# img = cv2.imread('whitegirl.jpg')
# img = cv2.resize(img, (224, 224))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# img = transform(img).cuda()
# img = img.unsqueeze(0)
#
# with torch.no_grad():
#     start = time.time()
#     out = model(img)
#     print("total time:{}".format(time.time() - start))
#     result = out.cpu().numpy()
#     # ind=np.argmax(out.cpu().numpy())
#     ind = np.argsort(result, axis=1)
#     for i in range(5):
#         print("predict:top {} = cls {} : score {}".format(i + 1, ind[0, 1000 - i - 1], result[0, 1000 - i - 1]))
#     print("done")

