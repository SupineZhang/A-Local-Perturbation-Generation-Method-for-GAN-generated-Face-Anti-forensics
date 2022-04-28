import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image

import cv2
import numpy as np
from pytorch_grad_cam import GuidedBackpropReLUModel
import models.resnet

import torch
import torch.nn as nn
import argparse
from PIL import Image
import torchvision.transforms as transforms
import gc

#single
def generate_mask(input_tensor, cam):

    target_category = 0
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam=torch.from_numpy(grayscale_cam).unsqueeze(1)
    cam_mask=grayscale_cam.expand(input_tensor.size())
    cam_soft_masks=torch.where(cam_mask < 0.5, torch.zeros(cam_mask.size()), cam_mask)

    cam_hard_masks=torch.where(cam_soft_masks >= 0.5, torch.ones(cam_soft_masks.size()), cam_soft_masks)

    return cam_soft_masks, cam_hard_masks


#ensemble
def get_cam_mask(input_tensor, resnetcam, xceptioncam, efficientcam):
   
        target_category = 0
        resnet_cam = resnetcam(input_tensor=input_tensor, target_category=target_category)
        efficient_cam = efficientcam(input_tensor=input_tensor, target_category=target_category)
        xception_cam = xceptioncam(input_tensor=input_tensor, target_category=target_category)
        
        resnet_cam = torch.from_numpy(resnet_cam)
        efficient_cam = torch.from_numpy(efficient_cam)
        xception_cam=torch.from_numpy(xception_cam)
        
        resnet_cam = torch.where(resnet_cam< 0.4, torch.zeros(resnet_cam.size()), resnet_cam)
        efficient_cam = torch.where(efficient_cam < 0.5, torch.zeros(efficient_cam.size()), efficient_cam)
        xception_cam = torch.where(xception_cam< 0.25, torch.zeros(xception_cam.size()), xception_cam)
        grayscale_cam = (resnet_cam+efficient_cam+xception_cam)/3
        grayscale_cam = grayscale_cam.unsqueeze(1)
        cam_mask = grayscale_cam.expand(input_tensor.size())
        cam_soft_masks = torch.where(cam_mask < 0.3, torch.zeros(cam_mask.size()), cam_mask)
        cam_hard_masks = torch.where(cam_soft_masks >= 0.3, torch.ones(cam_mask.size()), cam_soft_masks)

        return cam_soft_masks, cam_hard_masks
        


















