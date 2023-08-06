import argparse
import itertools
from itertools import zip_longest as zip
import tkinter
import matplotlib
import torchvision
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import torch
import IIC.code.archs as archs
from IIC.code.utils.segmentation.data import segmentation_create_dataloaders
import os
import cv2
import os.path as osp
import numpy as np
from PIL import Image


# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--opt", type=str, default="Adam")
parser.add_argument("--mode", type=str, default="IID")  # or IID+

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--dataset_root", type=str, required=True)

parser.add_argument("--use_coarse_labels", default=False,
                    action="store_true")  # COCO, Potsdam
parser.add_argument("--fine_to_coarse_dict", type=str,  # COCO #my_change
                    default="E:/MASTER/Uni/Term4/IIC_code/IIC/code/datasets"
                            "/segmentation/util/out/fine_to_coarse_dict.pickle")

parser.add_argument("--include_things_labels", default=False,
                    action="store_true")  # COCO
parser.add_argument("--incl_animal_things", default=False,
                    action="store_true")  # COCO
parser.add_argument("--coco_164k_curated_version", type=int, default=-1)  # COCO

parser.add_argument("--gt_k", type=int, required=True)
parser.add_argument("--output_k_A", type=int, required=True)
parser.add_argument("--output_k_B", type=int, required=True)

parser.add_argument("--lamb_A", type=float, default=1.0)
parser.add_argument("--lamb_B", type=float, default=1.0)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--lr_mult", type=float, default=0.1)

parser.add_argument("--use_uncollapsed_loss", default=False,
                    action="store_true")
parser.add_argument("--mask_input", default=False, action="store_true")

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_sz", type=int, required=True)  # num pairs
parser.add_argument("--num_dataloaders", type=int, default=3)
parser.add_argument("--num_sub_heads", type=int, default=5)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")
parser.add_argument("--restart", default=False, action="store_true")

parser.add_argument("--save_freq", type=int, default=5)
parser.add_argument("--test_code", default=False, action="store_true")

parser.add_argument("--head_B_first", default=False, action="store_true")
parser.add_argument("--batchnorm_track", default=False, action="store_true")

# data transforms
parser.add_argument("--no_sobel", default=False, action="store_true")

parser.add_argument("--include_rgb", default=False, action="store_true")
parser.add_argument("--pre_scale_all", default=False,
                    action="store_true")  # new
parser.add_argument("--pre_scale_factor", type=float, default=0.5)  #

parser.add_argument("--input_sz", type=int, default=161)  # half of kazuto1011

parser.add_argument("--use_random_scale", default=False,
                    action="store_true")  # new
parser.add_argument("--scale_min", type=float, default=0.6)
parser.add_argument("--scale_max", type=float, default=1.4)

# transforms we learn invariance to
parser.add_argument("--jitter_brightness", type=float, default=0.4)
parser.add_argument("--jitter_contrast", type=float, default=0.4)
parser.add_argument("--jitter_saturation", type=float, default=0.4)
parser.add_argument("--jitter_hue", type=float, default=0.125)

parser.add_argument("--flip_p", type=float, default=0.5)

parser.add_argument("--use_random_affine", default=False,
                    action="store_true")  # new
parser.add_argument("--aff_min_rot", type=float, default=-30.)  # degrees
parser.add_argument("--aff_max_rot", type=float, default=30.)  # degrees
parser.add_argument("--aff_min_shear", type=float, default=-10.)  # degrees
parser.add_argument("--aff_max_shear", type=float, default=10.)  # degrees
parser.add_argument("--aff_min_scale", type=float, default=0.8)
parser.add_argument("--aff_max_scale", type=float, default=1.2)

# local spatial invariance. Dense means done convolutionally. Sparse means done
#  once in data augmentation phase. These are not mutually exclusive
parser.add_argument("--half_T_side_dense", type=int, default=0)
parser.add_argument("--half_T_side_sparse_min", type=int, default=0)
parser.add_argument("--half_T_side_sparse_max", type=int, default=0)
config = parser.parse_args()
config.out_dir = os.path.join(config.out_root, str(config.model_ind))
config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)

torch.cuda.empty_cache()


def load_model():
    # net = archs.__dict__["SegmentationNet10aTwoHead"](config)
    net = archs.__dict__[config.arch](config)
    net = torch.nn.DataParallel(net)  # extra
    dict = torch.load("models/models/555/best_net.pytorch")
    net.load_state_dict(dict, strict=False)
    print("net type: ", type(net))
    # net.train()
    _,_, mapping_test_dataloader = segmentation_create_dataloaders(config)
    dataloader = mapping_test_dataloader

    for b_i, batch in enumerate(dataloader):
        imgs, flat_targets, mask= batch
        # print("batch type: ", type(batch))  # list
        # print("imgs type: ", type(imgs))  # torch.Tensor
        # print("flat_targets type: ", type(flat_targets))  # torch.Tensor
        # print("mask type: ", type(mask))  # torch.Tensor

        # print("batch len: ", len(batch))  # 3
        # print("imgs len: ", imgs.size()) #[120, 4, 128, 128]
        # print("flat_targets len: ", flat_targets.size()) #[120, 128, 128])
        # print("mask len: ", mask.size()) #[120, 128, 128]

        # print("imgs size: ", imgs.size()) #[120, 4, 128, 128]
        # print("imgs2 size: ", imgs[:, 0:3, :, :].size())

        # print("mask")
        # print(mask[0])
        #
        # print("test target")
        # print(flat_targets[0])
        #
        # print("imgs")
        # print(imgs[0])


##########OUTPUT###################################################
        np.set_printoptions(threshold=sys.maxsize)
        imgs=imgs[:, 0:3, :, :]  # torch.Size([120, 3, 128, 128])
        x_outs = net(imgs)

        # print("x_outs size ",len(x_outs))  # list_size_"1"
        # print("x_outs0 size ",x_outs[0].size())  # [120, 3, 128, 128]
        # print("x_outs0,0 size ",x_outs[0][1].size())  # second tensor [3, 128, 128]
        # print("x_outs tensor") #******important
        # print(x_outs[0][1]) #******important

        i=1
        output=x_outs[0][i].cpu().detach().numpy()  # class 'numpy.ndarray'
        out_image = np.argmax(output, axis=0)  # Convert probabilities to class labels
        out_image = np.where(out_image == 2, 255, out_image)  # Set background class to 0
        # out_image = np.where(out_image == 1, 130, out_image)  # Set middle class to 130
        out_image = np.where(out_image == 0, 0, out_image)  # Set foreground class to 255
        # print("out_image: ")
        # print(out_image)

        trans = torchvision.transforms.ToPILImage()
        outi = trans(imgs[i])
        outi.show()

        tensor_mask = (mask[i].cpu().detach().numpy())*255
        tensor_mask = Image.fromarray(np.uint8(tensor_mask), 'L')
        tensor_mask.show()

        tensor_image = out_image
        tensor_image = Image.fromarray(np.uint8(tensor_image), 'L')
        tensor_image.show()




######################INPUT CHANNELS################################
        # out1 = trans(imgs[0][0, :, :])
        # out1.show()
        #
        # out2 = trans(imgs[0][1, :, :])
        # out2.show()
        #
        # out3 = trans(imgs[0][2, :, :])
        # out3.show()
        #
        # out3 = trans(imgs[0][3, :, :])
        # out3.show()

######################OUTPUT CHANNELS################################
        # out1 = trans(x_outs[0][1][0])
        # out1.show()
        #
        # out2 = trans(x_outs[0][1][1])
        # out2.show()
        #
        # out3 = trans(x_outs[0][1][2])
        # out3.show()




        print("End")
        print("x_outs ",x_outs.size())













if __name__ == '__main__':
    load_model()

