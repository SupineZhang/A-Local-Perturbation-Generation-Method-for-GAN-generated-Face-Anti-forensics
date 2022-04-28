import cv2
import os
import numpy as np
import argparse
import torch
import kornia
import xlwt
import lpips

from torchvision import transforms

from utils import read_data, clean_save, adv_save, perturbation_save

from models.resnet import resnet50 as create_resnet
from models.xception import xception as create_xception
from models.modelEfficientNet import efficientnet_b0 as create_efficient
from models.densenet import densenet121 as create_dense
import models.gramnet as gramnet
from models.DiscNet import discnet as create_disc
from models.AlexNet import alexnet as create_alex
from models.Mesonet import meso as create_meso






import model

from data_preprocessing import data_preprocess
from tqdm import tqdm
from cam_module import generate_mask, get_cam_mask
from pytorch_grad_cam import GradCAM

save_xls='/data/zht/ensemble_stylegan/test.xls'
aa=2
bb=3

def main(args):
    gen_input_nc=args.image_num_channels
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    if not os.path.exists(args.save_image_path):
        os.makedirs(args.save_image_path)

    workbook = xlwt.Workbook(encoding='utf-8')
    # worksheet
    similarity = workbook.add_sheet('SSIM_PSNR_LPIPS_L2_MSE')
    accuracy=workbook.add_sheet('ClnAcc_AdvAcc')

    # loading models
    resnet50_model = create_resnet(num_classes=args.model_num_classes).to(device)
    resnet50_model.load_state_dict(torch.load(args.pretrained_resnet50_path,map_location=device))
    resnet50_model.eval()

    xception_model = create_xception(num_classes=args.model_num_classes).to(device)
    xception_model.load_state_dict(torch.load(args.pretrained_xception_path, map_location=device))
    xception_model.eval()

    efficient_model = create_efficient(num_classes=args.model_num_classes).to(device)
    efficient_model.load_state_dict(torch.load(args.pretrained_efficient_path, map_location=device))
    efficient_model.eval()

    densenet_model = create_dense(num_classes=args.model_num_classes).to(device)
    densenet_model.load_state_dict(torch.load(args.pretrained_densenet_path, map_location=device))
    densenet_model.eval()

    alexnet_model = create_alex().to(device)
    alexnet_model.load_state_dict(torch.load(args.pretrained_alexnet_path, map_location=device))
    alexnet_model.eval()

    discnet_model = create_disc().to(device)
    discnet_model.load_state_dict(torch.load(args.pretrained_discnet_path, map_location=device))
    discnet_model.eval()

    meso_model = create_meso().to(device)
    meso_model.load_state_dict(torch.load(args.pretrained_mesonet_path, map_location=device))
    meso_model.eval()

    gramnet_model = gramnet.resnet18().to(device)
    gramnet_model.load_state_dict(torch.load(args.pretrained_gramnet_path, map_location=device))
    gramnet_model.eval()

    rfm = create_xception(num_classes=args.model_num_classes).to(device)
    rfm.load_state_dict(torch.load(args.pretrained_rfm_path, map_location=device))
    rfm.eval()



    #loading generator
    pretrained_G = model.Generator(gen_input_nc, args.image_num_channels).to(device)
    pretrained_G.load_state_dict(torch.load(args.pretrained_generator_path,map_location=device))
    pretrained_G.eval()

    test_images_path, test_images_label = read_data(args.test_path)
    test_data_set, test_loader = data_preprocess(test_images_path, test_images_label, args.image_size, args.batch_size)

    #GradCAM
    resnet50_cam = GradCAM(model=resnet50_model, target_layers=[resnet50_model.layer4[-1]])
    xception_cam = GradCAM(model=xception_model, target_layers=[xception_model.conv4])
    efficient_cam = GradCAM(model=efficient_model, target_layers=[efficient_model.features[-1]])

    #lpips
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    #SSIm
    ssim = kornia.metrics.SSIM(11)
    loss_mse = torch.nn.MSELoss()
    #save
    ssim_sum = 0
    psnr_sum = 0
    lpips_alex_sum = 0
    lpips_vgg_sum = 0
    l2_sum = 0
    mse_sum=0

    resnet_num_correct = 0 #adv
    xception_num_correct = 0
    efficient_num_correct = 0
    densenet_num_correct = 0
    gramnet_num_correct = 0
    discnet_num_correct=0
    alexnet_num_correct = 0
    mesonet_num_correct = 0
    rfm_num_correct=0

    resnet_num_correct0 = 0 #clean
    xception_num_correct0 = 0
    efficient_num_correct0 = 0
    densenet_num_correct0 = 0
    gramnet_num_correct0 = 0
    discnet_num_correct0=0
    alexnet_num_correct0 = 0
    mesonet_num_correct0 = 0
    rfm_num_correct0=0


    #testing
    for i, data in enumerate(tqdm(test_loader), 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        perturbation = pretrained_G(test_img)
        cam_soft_masks, cam_masks = get_cam_mask(test_img, resnet50_cam, xception_cam, efficient_cam)
        perturbation = torch.mul(perturbation, cam_masks.to(device))
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        noise=adv_img-test_img

        # #save image
        # # clean_save(test_img, args.save_image_path, i)
        # adv_save(adv_img, args.save_image_path, i)
        # # perturbation_save(noise, args.save_image_path, i)

        pred_lab = torch.argmax(resnet50_model(test_img), 1)
        resnet_num_correct0 += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(xception_model(test_img), 1)
        xception_num_correct0 += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(efficient_model(test_img), 1)
        efficient_num_correct0 += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(densenet_model(test_img), 1)
        densenet_num_correct0 += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(gramnet_model(test_img), 1)
        gramnet_num_correct0 += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(discnet_model(test_img), 1)
        discnet_num_correct0 += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(alexnet_model(test_img), 1)
        alexnet_num_correct0 += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(meso_model(test_img), 1)
        mesonet_num_correct0 += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(rfm(test_img), 1)
        rfm_num_correct0 += torch.sum(pred_lab == test_label, 0)


        pred_lab = torch.argmax(resnet50_model(adv_img), 1)
        resnet_num_correct += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(xception_model(adv_img), 1)
        xception_num_correct += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(efficient_model(adv_img), 1)
        efficient_num_correct += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(densenet_model(adv_img), 1)
        densenet_num_correct += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(gramnet_model(adv_img), 1)
        gramnet_num_correct += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(discnet_model(adv_img), 1)
        discnet_num_correct += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(alexnet_model(adv_img), 1)
        alexnet_num_correct += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(meso_model(adv_img), 1)
        mesonet_num_correct += torch.sum(pred_lab == test_label, 0)
        pred_lab = torch.argmax(rfm(adv_img), 1)
        rfm_num_correct += torch.sum(pred_lab == test_label, 0)

        psnr_batch = kornia.losses.psnr_loss(test_img, adv_img, max_val=2).item()
        ssim_batch = torch.mean(ssim(test_img, adv_img)).item()
        lpips_alex_batch = torch.mean(loss_fn_alex(test_img, adv_img)).item()
        lpips_vgg_batch = torch.mean(loss_fn_vgg(test_img, adv_img)).item()
        l2_batch = torch.mean(torch.norm(noise.view(noise.shape[0], -1), 2, dim=1)).item()
        mse_batch = loss_mse(test_img, adv_img).item()

        psnr_sum += psnr_batch
        ssim_sum += ssim_batch
        lpips_alex_sum += lpips_alex_batch
        lpips_vgg_sum += lpips_vgg_batch
        l2_sum += l2_batch
        mse_sum+=mse_batch

    # print('accuracy of adv imgs in resnet: %f\n' % (resnet_num_correct.item() / len(test_data_set)))
    # print('accuracy of adv imgs in xception: %f\n' % (xception_num_correct.item() / len(test_data_set)))
    # print('accuracy of adv imgs in efficient: %f\n' % (efficient_num_correct.item() / len(test_data_set)))
    # print('accuracy of adv imgs in densenet: %f\n' % (densenet_num_correct.item() / len(test_data_set)))
    # print('accuracy of adv imgs in gramnet: %f\n' % (gramnet_num_correct.item() / len(test_data_set)))
    # print('accuracy of adv imgs in discnet: %f\n' % (discnet_num_correct.item() / len(test_data_set)))
    accuracy.write(1, 1, label='cam,l2=1,kl,0vgg,ce')
    accuracy.write(aa, 1, label=str(resnet_num_correct0.item() / len(test_data_set)))
    accuracy.write(aa, 2, label=str(xception_num_correct0.item() / len(test_data_set)))
    accuracy.write(aa, 3, label=str(efficient_num_correct0.item() / len(test_data_set)))
    accuracy.write(aa, 4, label=str(densenet_num_correct0.item() / len(test_data_set)))
    accuracy.write(aa, 8, label=str(gramnet_num_correct0.item() / len(test_data_set)))
    accuracy.write(aa, 6, label=str(discnet_num_correct0.item() / len(test_data_set)))
    accuracy.write(aa, 5, label=str(alexnet_num_correct0.item() / len(test_data_set)))
    accuracy.write(aa, 7, label=str(mesonet_num_correct0.item() / len(test_data_set)))
    accuracy.write(aa, 9, label=str(rfm_num_correct0.item() / len(test_data_set)))

    accuracy.write(bb, 1, label=str(resnet_num_correct.item() / len(test_data_set)))
    accuracy.write(bb, 2, label=str(xception_num_correct.item() / len(test_data_set)))
    accuracy.write(bb, 3, label=str(efficient_num_correct.item() / len(test_data_set)))
    accuracy.write(bb, 4, label=str(densenet_num_correct.item() / len(test_data_set)))
    accuracy.write(bb, 8, label=str(gramnet_num_correct.item() / len(test_data_set)))
    accuracy.write(bb, 6, label=str(discnet_num_correct.item() / len(test_data_set)))
    accuracy.write(bb, 5, label=str(alexnet_num_correct.item() / len(test_data_set)))
    accuracy.write(bb, 7, label=str(mesonet_num_correct.item() / len(test_data_set)))
    accuracy.write(bb, 9, label=str(rfm_num_correct.item() / len(test_data_set)))

    similarity.write(0, 1, label='cam,l2=1,kl,0vgg,ce')
    similarity .write(1, 1, label=str((ssim_sum / len(test_loader))))
    similarity .write(1, 2, label=str((psnr_sum / len(test_loader))))
    similarity .write(1, 3, label=str((lpips_alex_sum / len(test_loader))))
    similarity .write(1, 4, label=str((lpips_vgg_sum / len(test_loader))))
    similarity .write(1, 5, label=str((l2_sum / len(test_loader))))
    similarity .write(1, 6, label=str((mse_sum / len(test_loader))))
    workbook.save(save_xls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-num-channels', type=int, default=3)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--model-num-classes', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weight-G-decay", type=float, default=0.0004)
    parser.add_argument("--weight-D-decay", type=float, default=0.0004)

    parser.add_argument('--data-path', type=str,
                        default="/data/zht/dataset/stylegan_real_dataset/train/celeba_stylegan")
    parser.add_argument('--test-path', type=str,
                        default="/data/zht/dataset/stylegan_real_dataset/test/celeba_stylegan")

    parser.add_argument('--pretrained-resnet50-path', type=str,
                        default='/data/zht/stylegan_detection/resnet/weights/resnet.pth',
                        help='initial weights path')
    parser.add_argument('--pretrained-xception-path', type=str,
                        default='/data/zht/stylegan_detection/xception/weights/xception.pth',
                        help='initial weights path')
    parser.add_argument('--pretrained-efficient-path', type=str,
                        default='/data/zht/stylegan_detection/efficientnet/weights/efficient.pth',
                        help='initial weights path')
    parser.add_argument('--pretrained-densenet-path', type=str,
                        default='/data/zht/stylegan_detection/densenet/weights/densenet.pth',
                        help='initial weights path')
    parser.add_argument('--pretrained-gramnet-path', type=str,
                        default='/data/zht/stylegan_detection/gramnet/weights/model.pth',
                        help='initial weights path')
    parser.add_argument('--pretrained-discnet-path', type=str,
                        default='/data/zht/stylegan_detection/discnet/weights/model.pth',
                        help='initial weights path')
    parser.add_argument('--pretrained-alexnet-path', type=str,
                        default='/data/zht/stylegan_detection/alexnet/weights/alexnet.pth',
                        help='initial weights path')
    parser.add_argument('--pretrained-mesonet-path', type=str,
                        default='/data/zht/stylegan_detection/mesonet/weights/meso.pth',
                        help='initial weights path')
    parser.add_argument('--pretrained-rfm-path', type=str,
                        default='/data/zht/stylegan_detection/RFM-main/xbase_xception_model.pth',
                        help='initial weights path')


    # parser.add_argument('--pretrained-generator-path', type=str,
    #                     default="/data/zht/ensemble_stylegan/baseline/generator/netG_epoch_1.pth")
    parser.add_argument('--pretrained-generator-path', type=str,
                        default="/data/zht/ensemble_stylegan/generator/netG.pth")
    parser.add_argument('--save-image-path', type=str,
                        default="/data/zht/results/adv/")

    parser.add_argument('--box-min', type=int, default=0)
    parser.add_argument('--box-max', type=int, default=1)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)


