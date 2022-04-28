import torch
import utils
from vgg import Vgg16
import torch.nn as nn

loss_mse = torch.nn.MSELoss()

def get_perceptual_loss(vgg, input, adv):
    # get vgg features
    x_features = vgg(input)
    adv_features = vgg(adv)
    # calculate style loss
    x_gram = [utils.gram(fmap) for fmap in x_features]
    adv_gram = [utils.gram(fmap) for fmap in adv_features]
    style_loss = 0.0
    for j in range(4):
        style_loss += loss_mse(x_gram[j], adv_gram[j])
    style_loss = style_loss

    # calculate content loss (h_relu_2_2)
    xcon = x_features[1]
    acon = adv_features[1]
    content_loss = loss_mse(xcon, acon)
    return style_loss, content_loss

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.epsilon = 1e-8  # the parameter to make sure the denominator non-zero

    def forward(self, map_pred,
                map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()

        map_pred = map_pred.view(1, -1)  # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1)  # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        # print("min1 and max1 are :", min1, max1)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon)  # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        # print("min2 and max2 are :", min2, max2)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon)  # min-max normalization for keeping KL loss non-NAN

        map_pred = map_pred / (
                    torch.sum(map_pred) + self.epsilon)  # normalization step to make sure that the map_pred sum to 1
        map_gtd = map_gtd / (
                    torch.sum(map_gtd) + self.epsilon)  # normalization step to make sure that the map_gtd sum to 1
        # print("map_pred is :", map_pred)
        # print("map_gtd is :", map_gtd)

        KL = torch.log(map_gtd / (map_pred + self.epsilon) + self.epsilon)
        # print("KL 1 is :", KL)
        KL = map_gtd * KL
        # print("KL 2 is :", KL)
        KL = torch.sum(KL)
        # print("KL 3 is :", KL)
        # print("KL loss is :", KL)

        return KL





















