import argparse
import torch

from utils import read_data
from grad_advGAN import AdvGAN_Attack
from models.resnet import resnet50 as create_resnet
from models.xception import xception as create_xception
from models.modelEfficientNet import efficientnet_b0 as create_efficient
from data_preprocessing import data_preprocess
from pytorch_grad_cam import GradCAM


save_path = '/data/zht/ensemble_stylegan/baseline1/loss.txt'

def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #dataset
    train_images_path, train_images_label = read_data(args.data_path)
    train_dataset,train_loader=data_preprocess(train_images_path, train_images_label,args.image_size, args.batch_size )

    # models
    resnet50_model = create_resnet(num_classes=args.num_classes).to(device)
    resnet50_model.load_state_dict(torch.load(args.pretrained_resnet50_path, map_location=device))
    resnet50_model.eval()

    xception_model = create_xception(num_classes=args.num_classes).to(device)
    xception_model.load_state_dict(torch.load(args.pretrained_xception_path, map_location=device))
    xception_model.eval()

    efficient_model = create_efficient(num_classes=args.num_classes).to(device)
    efficient_model.load_state_dict(torch.load(args.pretrained_efficient_path, map_location=device))
    efficient_model.eval()

    # gradcam
    resnet50_cam = GradCAM(model=resnet50_model, target_layers=[resnet50_model.layer4[-1]])
    xception_cam = GradCAM(model=xception_model, target_layers=[xception_model.conv4])
    efficient_cam = GradCAM(model=efficient_model, target_layers=[efficient_model.features[-1]])


    #attack based on advGAN
    advGAN = AdvGAN_Attack(device,
                           resnet50_model,
                           efficient_model,
                           xception_model,
                           args.num_classes,
                           args.image_num_channels,
                           args.box_min,
                           args.box_max,
                           args.l_inf_bound,
                           args.lr,
                           args.b1,
                           args.b2,
                           args.alpha,
                           args.beta,
                           args.gamma,
                           args.c,
                           args.n_steps_D,
                           args.n_steps_G,
                           resnet50_cam,
                           efficient_cam,
                           xception_cam)

    advGAN.train(train_loader, args.epochs, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-num-channels', type=int, default=3)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--weight-G-decay", type=float, default=0.0004)
    # parser.add_argument("--weight-D-decay", type=float, default=0.0004)

    parser.add_argument('--data-path', type=str,
                        default="/data/zht/dataset/stylegan_real_dataset/train/celeba_stylegan")
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
                        default='/data/zht/stylegan_detection/gramnet/weights/gramenet.pth',
                        help='initial weights path')

    parser.add_argument('--pretrained-generator-path', type=str,
                        default="")

    parser.add_argument('--box-min', type=int, default=0)
    parser.add_argument('--box-max', type=int, default=1)
    parser.add_argument('--alpha', type=int, default=10)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--gamma', type=int, default=1)
    parser.add_argument('--c', type=float, default=0)
    parser.add_argument('--n-steps-D', type=int, default=1)
    parser.add_argument('--n-steps-G', type=int, default=1)
    parser.add_argument('--l-inf-bound', type=float, default=0.3)


    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

