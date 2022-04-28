import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import utils,transforms
import numpy as np
from PIL import Image
import cv2

import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


class dist_average:
    def __init__(self, local_rank):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = local_rank
        self.acc = torch.zeros(1).to(local_rank)
        self.count = 0

    def step(self, input_):
        self.count += 1
        if type(input_) != torch.Tensor:
            input_ = torch.tensor(input_).to(self.local_rank, dtype=torch.float)
        else:
            input_ = input_.detach()
        self.acc += input_

    def get(self):
        dist.all_reduce(self.acc, op=dist.ReduceOp.SUM)
        self.acc /= self.world_size
        return self.acc.item() / self.count


def ACC(x, y):
    with torch.no_grad():
        a = torch.max(x, dim=1)[1]
        acc = torch.sum(a == y).float() / x.shape[0]
    # print(y,a,acc)
    return acc


def cont_grad(x, rate=1):
    return rate * x + (1 - rate) * x.detach()



def read_data(root: str):

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF"]  # 支持的文件后缀类型

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    class_dicts = dict([val, key] for key, val in class_indict.items())


    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    # 获取该类别对应的索引
    image_class = int(class_dicts['celeba_stylegan'])
    # 记录该类别的样本数量
    every_class_num.append(len(images))

    for img_path in images:
        train_images_path.append(img_path)
        train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))

    return train_images_path, train_images_label


#以Image的格式进行存储
def imageSavePIL(images,fileName,normalization=True,mean=0,std=1):
    image=utils.make_grid(images)
    #是否原图进行了normalization
    if normalization:
        #交换之后，(H,W,C)
        image=image.permute(1,2,0)
        image=(image*torch.tensor(std)+torch.tensor(mean))
        #交换之后,(C,H,W)
        image=image.permute(2,0,1)
    #将tensor转化为Image格式
    image=transforms.ToPILImage()(image)
    #存储图片
    image.save(fileName)

#用plt库进行存储
def imageSavePLT(images,fileName,normalization=False,mean=0,std=1):
    image = utils.make_grid(images)
    #交换维度之后 (H,W,C)
    image = image.permute(1,2,0).cpu().numpy()
    if normalization:
        image=(image.cpu()*torch.tensor(std)+torch.tensor(mean)).numpy()
    #存储图片
    plt.imsave(fileName,image)

def save_image(images, filename):
    image_tensor = utils.make_grid(images)
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)

#用plt库进行存储
def clean_save(images,fileName,i):
    # image = utils.make_grid(images)
    for j in range(images.size(0)):
        image = images[j, :, :, :]
        utils.save_image(image, fileName + str(i)+"f_" + str(j) + ".png")


def adv_save(images, fileName, i):
    # image = utils.make_grid(images)
    for j in range(images.size(0)):
        image = images[j, :, :, :]
        utils.save_image(image, fileName + str(i) + "a_" + str(j) + ".png")


#使用opencv来保存
def perturbation_save(perturbation,filename,i):
    # for i in  range(perturbation.size(0)):
    for j in range(perturbation.size(0)):
        image = perturbation[j, :, :, :]
        image = image.permute(1, 2, 0).detach().cpu().numpy()
        cv2.imwrite(filename+ str(i) + "p_" + str(j) + ".png", image * 255)


# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        # outputs = []
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)
            x = module(x)
            if name == self.extracted_layers:
                break
        return x








