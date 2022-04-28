import os
import torch
from torchvision import transforms

from my_dataset import MyDataSet



def data_preprocess(images_path, images_label,image_size=128,  batch_size=64,):

    # 处理图片大小
    data_transform =transforms.Compose([transforms.Resize(image_size),
                                     transforms.CenterCrop(image_size),
                                     transforms.ToTensor(),
                                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

    # 实例化训练数据集
    data_set = MyDataSet(images_path=images_path,
                         images_class=images_label,
                         transform=data_transform)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    data_loader = torch.utils.data.DataLoader(data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw, )
    return data_set, data_loader
