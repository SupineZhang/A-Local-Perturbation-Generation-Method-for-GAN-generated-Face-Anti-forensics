import matplotlib
matplotlib.use('Agg')

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import model
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import os
from tqdm import tqdm
from cam_module import generate_mask, get_cam_mask
from pytorch_grad_cam import GradCAM
import utils
from vgg import Vgg16
from loss import get_perceptual_loss as vggloss
from loss import KLLoss
import torchextractor as tx
from utils import FeatureExtractor


models_G_path = './baseline/generator/'
losses_path = './baseline/losses/'
STYLE_WEIGHT=1e5
CONTENT_WEIGHT=1e0
d=1e-8
loss_mse = torch.nn.MSELoss()


#weights initialize
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class AdvGAN_Attack:
    def __init__(self,
                 device,
                 resnet_model,
                 efficient_model,
                 xception_model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max,
                 l_inf_bound,
                 lr,
                 b1,
                 b2,
                 alpha,
                 beta,
                 gamma,
                 c,
                 n_steps_D,
                 n_steps_G,
                 resnet_cam,
                 efficient_cam,
                 xception_cam):

        self.device = device
        self.model_num_labels = model_num_labels
        self.resnet_model = resnet_model
        self.efficient_model = efficient_model
        self.xception_model = xception_model


        self.lr=lr
        self.b1=b1
        self.b2=b2

        self.l_inf_bound = l_inf_bound
        #box constraint
        self.box_min = box_min
        self.box_max = box_max
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c = c

        self.n_steps_D = n_steps_D
        self.n_steps_G = n_steps_G

        self.resnet_cam = resnet_cam
        self.efficient_cam=efficient_cam
        self.xception_cam=xception_cam

        self.netG = model.Generator(image_nc, image_nc).to(device)
        self.netDisc = model.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=self.lr, betas=(b1,b2))
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=self.lr, betas=(b1,b2))
        # self.bce_loss=nn.BCELoss()
        self.celoss =nn.CrossEntropyLoss()
        self.bceloss=nn.BCEWithLogitsLoss()
        self.klloss=KLLoss()

        self.vgg=Vgg16().to(device)


        if not os.path.exists(models_G_path):
            os.makedirs(models_G_path)
        if not os.path.exists(losses_path):
            os.makedirs(losses_path)


    def train_batch(self, x, labels):

        # optimize D
        for i in range(self.n_steps_D):
            #生成器生成扰动
            perturbation = self.netG(x)
            cam_soft_masks,cam_hard_masks= get_cam_mask(x, self.resnet_cam, self.xception_cam, self.efficient_cam)
            perturbation = torch.mul(perturbation, cam_hard_masks.to(self.device))
            perturbation=torch.clamp(perturbation, -self.l_inf_bound, self.l_inf_bound)
            adv_images = perturbation + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()

            logits_real, pred_real = self.netDisc(x)
            logits_fake,pred_fake = self.netDisc(adv_images.detach())
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_GAN = loss_D_fake + loss_D_real
            loss_D_GAN.backward()

            self.optimizer_D.step()

        # optimize G
        for i in range(self.n_steps_G):
            self.netG.zero_grad()

            # cal G's loss in GAN
            logits_fake, pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_perturb = torch.mean(torch.norm(torch.mul(perturbation, (1-cam_soft_masks).to(self.device)).view(perturbation.shape[0], -1), 2, dim=1))
            loss_perturb = torch.max(loss_perturb - self.c, torch.zeros(1, device=self.device))

            #vgg loss
            loss_style, loss_content = vggloss(self.vgg, x, adv_images)
            loss_vgg= STYLE_WEIGHT*loss_style + CONTENT_WEIGHT*loss_content

            # ce loss
            logits_resnet = self.resnet_model(adv_images)
            logits_xception = self.xception_model(adv_images)
            logits_efficient = self.efficient_model(adv_images)

            onehot_labels = torch.eye(2, device=self.device)[torch.ones_like(labels, device=self.device)]
            loss_ce = (self.bceloss(logits_resnet, onehot_labels)+\
                      self.bceloss(logits_xception, onehot_labels)+\
                      self.bceloss(logits_efficient, onehot_labels))/3

            #latent kl loss
            features_resnet_x = self.resnet_model.getfeatures(x)
            features_resnet_adv = self.resnet_model.getfeatures(adv_images)
            features_xception_x = self.xception_model.getfeatures(x)
            features_xception_adv = self.xception_model.getfeatures(adv_images)
            features_efficient_x = self.efficient_model.getfeatures(x)
            features_efficient_adv = self.efficient_model.getfeatures(adv_images)
            loss_latent = 3 / (self.klloss(features_resnet_adv, features_resnet_x) +
                               self.klloss(features_xception_adv, features_xception_x) +
                               self.klloss(features_efficient_adv, features_efficient_x))
            loss_adv=loss_ce
            


            loss_G = self.gamma*loss_G_fake + self.alpha * loss_adv + self.beta * loss_perturb + loss_vgg +loss_latent

            loss_G.backward()
            self.optimizer_G.step()
        return loss_D_GAN.item(), loss_G.item(), loss_adv.item(), loss_G_fake.item(), loss_perturb.item(), loss_vgg.item(), loss_ce.item(), loss_latent.item()

    def train(self, train_dataloader, epochs, save_path):
        tb_writer = SummaryWriter()
        loss_D, loss_G, loss_adv, loss_G_gan, loss_hinge, loss_vgg, loss_ce, loss_latent= [], [], [], [], [], [], [], []
        for epoch in (range(1, epochs+1)):
            train_dataloader=tqdm(train_dataloader)
            loss_D_sum = 0
            loss_G_sum=0
            loss_adv_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_ce_sum=0
            loss_latent_sum=0
            loss_vgg_sum=0

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_batch, loss_adv_batch, loss_G_fake_batch, loss_perturb_batch, loss_vgg_batch, loss_ce_batch, loss_latent_batch= \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_sum +=loss_G_batch
                loss_adv_sum += loss_adv_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_vgg_sum += loss_vgg_batch
                loss_ce_sum += loss_ce_batch
                loss_latent_sum += loss_latent_batch



            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_vgg: %.3f\nloss_adv:%.3f, loss_ce:%.3f, loss_latent:%.3f\n" %
                  (epoch, loss_D_sum/num_batch, loss_G_sum/num_batch,loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_vgg_sum/num_batch, loss_adv_sum/num_batch, loss_ce_sum/num_batch, loss_latent_sum/num_batch))
            # 记录 avg_loss 和 epoch acc
            with open(save_path, "a") as f:
                f.write("[E: %d] loss_D: %.3f, loss_G:%.3f, loss_G_fake: %.3f,  loss_perturb: %.3f , loss_vgg: %.3f, loss_adv: %.3f, loss_ce:%.3f, loss_latent:%.3f\n" % (
                    epoch, loss_D_sum/num_batch, loss_G_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch,  loss_vgg_sum/num_batch, loss_adv_sum/num_batch, loss_ce_sum/num_batch,loss_latent_sum/num_batch))
                f.write("\n")
            loss_D.append(loss_D_sum / num_batch)
            loss_G.append(loss_G_sum / num_batch)
            loss_adv.append(loss_adv_sum / num_batch)
            loss_G_gan.append(loss_G_fake_sum / num_batch)
            loss_hinge.append(loss_perturb_sum / num_batch)
            loss_ce.append(loss_ce_sum / num_batch)
            loss_latent.append(loss_latent_sum / num_batch)
            loss_vgg.append(loss_vgg_sum/num_batch)

            tags = ["loss_D", "loss_G", "loss_G-fake", "loss_perturb", "loss_vgg", "loss_adv", "loss_ce", "loss_latent"]
            tb_writer.add_scalar(tags[0], loss_D_sum/num_batch, epoch)
            tb_writer.add_scalar(tags[1], loss_G_sum / num_batch, epoch)
            tb_writer.add_scalar(tags[2], loss_G_fake_sum / num_batch, epoch)
            tb_writer.add_scalar(tags[3], loss_perturb_sum / num_batch, epoch)
            tb_writer.add_scalar(tags[4], loss_vgg_sum / num_batch, epoch)
            tb_writer.add_scalar(tags[5], loss_adv_sum / num_batch, epoch)
            tb_writer.add_scalar(tags[6], loss_ce_sum / num_batch, epoch)
            tb_writer.add_scalar(tags[7], loss_latent_sum / num_batch, epoch)

            netG_file_name = models_G_path + 'netG_epoch_' + str(epoch) + '.pth'
            torch.save(self.netG.state_dict(), netG_file_name)


        plt.figure()
        plt.plot(loss_D)
        plt.savefig(losses_path + '/loss_D.png')

        plt.figure()
        plt.plot(loss_G)
        plt.savefig(losses_path + '/loss_G.png')

        plt.figure()
        plt.plot(loss_adv)
        plt.savefig(losses_path + '/loss_adv.png')

        plt.figure()
        plt.plot(loss_G_gan)
        plt.savefig(losses_path + '/loss_G_gan.png')

        plt.figure()
        plt.plot(loss_hinge)
        plt.savefig(losses_path + '/loss_hinge.png')

        plt.figure()
        plt.plot(loss_vgg)
        plt.savefig(losses_path + '/loss_vgg.png')


        plt.figure()
        plt.plot(loss_ce)
        plt.savefig(losses_path +'/loss_ce.png')

        plt.figure()
        plt.plot(loss_latent)
        plt.savefig(losses_path + '/loss_latent.png')

