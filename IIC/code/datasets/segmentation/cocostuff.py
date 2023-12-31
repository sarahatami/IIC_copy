# based on
# https://github.com/kazuto1011/deeplab-pytorch/blob/master/libs/datasets
# /cocostuff.py

from __future__ import print_function
import os
import os.path as osp
import pickle
from glob import glob

import cv2

import numpy as np
import scipy.io as sio
import torch
import torchvision
import torchvision.transforms as tvt
from PIL import Image
from torch.utils import data

from .util import cocostuff_fine_to_coarse
from .util.cocostuff_fine_to_coarse import generate_fine_to_coarse
from ...utils.segmentation.render import render
from ...utils.segmentation.transforms import \
    pad_and_or_crop, random_affine, custom_greyscale_numpy

__all__ = ["Coco164kCuratedFew"]

RENDER_DATA = False


class _Coco(data.Dataset):
    """Base class
  This contains fields and methods common to all COCO datasets:
  (COCO-fine) (182)
  COCO-coarse (27)
  COCO-few (6)
  (COCOStuff-fine) (91)
  COCOStuff-coarse (15)
  COCOStuff-few (3)

  For both 10k and 164k (though latter is unimplemented)
  """

    def __init__(self, config=None, split=None, purpose=None, preload=False):
        super(_Coco, self).__init__()

        self.split = split  # specified in data.py
        self.purpose = purpose  # specified in data.py

        self.root = config.dataset_root

        self.single_mode = hasattr(config, "single_mode") and config.single_mode

        # always used (labels fields used to make relevancy mask for train)
        self.gt_k = config.gt_k
        self.pre_scale_all = config.pre_scale_all
        self.pre_scale_factor = config.pre_scale_factor
        self.input_sz = config.input_sz

        self.include_rgb = config.include_rgb
        self.no_sobel = config.no_sobel

        assert ((not hasattr(config, "mask_input")) or (not config.mask_input))
        self.mask_input = False

        # only used if purpose is train
        if purpose == "train":
            self.use_random_scale = config.use_random_scale
            if self.use_random_scale:
                self.scale_max = config.scale_max
                self.scale_min = config.scale_min

            self.jitter_tf = tvt.ColorJitter(brightness=config.jitter_brightness,
                                             contrast=config.jitter_contrast,
                                             saturation=config.jitter_saturation,
                                             hue=config.jitter_hue)

            self.flip_p = config.flip_p  # 0.5

            self.use_random_affine = config.use_random_affine
            if self.use_random_affine:
                self.aff_min_rot = config.aff_min_rot
                self.aff_max_rot = config.aff_max_rot
                self.aff_min_shear = config.aff_min_shear
                self.aff_max_shear = config.aff_max_shear
                self.aff_min_scale = config.aff_min_scale
                self.aff_max_scale = config.aff_max_scale

        assert (not preload)

        self.files = []
        # self.images = []
        # self.labels = []

        if not osp.exists(config.fine_to_coarse_dict):  # False
            generate_fine_to_coarse(config.fine_to_coarse_dict)

        with open(config.fine_to_coarse_dict, "rb") as dict_f:
            d = pickle.load(dict_f)  # dict
            self._fine_to_coarse_dict = d["fine_index_to_coarse_index"]  # index(0-181):index(0-26)

        cv2.setNumThreads(0)

    def _prepare_train(self, index, img, label):
        # This returns gpu tensors.
        # label is passed in canonical [0 ... 181] indexing

        # print("img.shape: ",img.shape)  # (400, 500, 3) or (482, 640, 3) or ...
        # print("label.shape: ",label.shape)  #  (400, 500) or (482, 640) or ...

        trans = torchvision.transforms.ToPILImage()
        # outi = trans(img)
        # outi.show()
        # outi = trans(label)
        # outi.show()

        assert (img.shape[:2] == label.shape)
        img = img.astype(np.float32)
        label = label.astype(np.int32)

        # shrink original images, for memory purposes, otherwise no point
        if self.pre_scale_all:  # True
            assert (self.pre_scale_factor < 1.)
            img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                             fy=self.pre_scale_factor,
                             interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize=None, fx=self.pre_scale_factor,
                               fy=self.pre_scale_factor,
                               interpolation=cv2.INTER_NEAREST)

        # basic augmentation transforms for both img1 and img2
        if self.use_random_scale:  # False
            # bilinear interp requires float img
            scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                           self.scale_min
            img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize=None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_NEAREST)

        # print("img.shape2",img.shape)  # (132, 165, 3) or (159, 211, 3) or ..
        # print("label.shape2",label.shape)  # (132, 165) or (159, 211) or ..


        # random crop to input sz
        img, coords = pad_and_or_crop(img, self.input_sz, mode="random")
        label, _ = pad_and_or_crop(label, self.input_sz, mode="fixed",
                                   coords=coords)
        # print("img.shape3",img.shape)  # (128, 128, 3)
        # print("label.shape3",label.shape)  # (128, 128)
        # print("label",label)  # from 0 to 148 or higher
        # outi = trans(label)
        # outi.show()

        _, mask_img1 = self._filter_label(label)  # True or False map
        # uint8 tensor as masks should be binary, also for consistency with prepare_train,
        # but converted to float32 in main loop because is used multiplicatively in loss

        mask_img1 = torch.from_numpy(mask_img1.astype(np.uint8)).cuda()  # 0 or 1 map

        # print('\n'.join([str(row) for row in mask_img1.tolist()]))
        # mask_img1 = np.where(mask_img1 == False, 0, mask_img1)
        # mask_img1 = np.where(mask_img1 == True, 255, mask_img1)
        # tensor_image = Image.fromarray(mask_img1.astype(np.uint8), 'L')
        # tensor_image.show()

        img1 = Image.fromarray(img.astype(np.uint8))

        # (img2) do jitter, no tf_mat change
        img2 = self.jitter_tf(img1)  # not in place, new memory, Photometric Transform:)
        img1 = np.array(img1)  # (128, 128, 3)
        img2 = np.array(img2)  # (128, 128, 3)

        # outi = trans(img1.astype('uint8'))
        # outi.show()
        # outi = trans(img2.astype('uint8'))
        # outi.show()
        ####################################################
        # channels still last
        if not self.no_sobel:
            img1 = custom_greyscale_numpy(img1, include_rgb=self.include_rgb)
            img2 = custom_greyscale_numpy(img2, include_rgb=self.include_rgb)

        img1 = img1.astype(np.float32) / 255.
        img2 = img2.astype(np.float32) / 255.

        # convert both to channel-first tensor format
        # make them all cuda tensors now, except label, for optimality
        img1 = torch.from_numpy(img1).permute(2, 0, 1).cuda()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).cuda()

        # mask if required
        if self.mask_input:  # False
            masked = 1 - mask_img1
            img1[:, masked] = 0
            img2[:, masked] = 0

        # (img2) do affine if nec, tf_mat changes
        if self.use_random_affine:  # False
            affine_kwargs = {"min_rot": self.aff_min_rot, "max_rot": self.aff_max_rot,
                             "min_shear": self.aff_min_shear,
                             "max_shear": self.aff_max_shear,
                             "min_scale": self.aff_min_scale,
                             "max_scale": self.aff_max_scale}
            img2, affine1_to_2, affine2_to_1 = random_affine(img2,
                                                             **affine_kwargs)  #
            # tensors
        else:
            affine2_to_1 = torch.zeros([2, 3]).to(torch.float32).cuda()  # identity
            affine2_to_1[0, 0] = 1
            affine2_to_1[1, 1] = 1
        # affine2_to_1:  tensor([[1., 0., 0.],
        #                        [0., 1., 0.]], device='cuda:0')

        # (img2) do random flip, tf_mat changes
        if np.random.rand() > self.flip_p:
            img2 = torch.flip(img2, dims=[2])  # horizontal, along width, geometric transform:)

            # applied affine, then flip, new = flip * affine * coord
            # (flip * affine)^-1 is just flip^-1 * affine^-1.
            # No order swap, unlike functions...
            # hence top row is negated
            affine2_to_1[0, :] *= -1.
            # affine2_to_1:  tensor([[-1., -0., -0.],
            #                        [0., 1., 0.]], device='cuda:0')

        if RENDER_DATA:
            render(img1, mode="image", name=("train_data_img1_%d" % index))
            render(img2, mode="image", name=("train_data_img2_%d" % index))
            render(affine2_to_1, mode="matrix",
                   name=("train_data_affine2to1_%d" % index))
            render(mask_img1, mode="mask", name=("train_data_mask_%d" % index))

        # outi = trans(img1)
        # outi.show()
        # outi = trans(mask_img1)
        # outi.show()
        # outi = trans(img2)
        # outi.show()

        return img1, img2, affine2_to_1, mask_img1

    def _prepare_train_single(self, index, img, label):
        # Returns one pair only, i.e. without transformed second image.
        # Used for standard CNN training (baselines).
        # This returns gpu tensors.
        # label is passed in canonical [0 ... 181] indexing

        assert (img.shape[:2] == label.shape)
        img = img.astype(np.float32)
        label = label.astype(np.int32)

        # shrink original images, for memory purposes, otherwise no point
        if self.pre_scale_all:
            assert (self.pre_scale_factor < 1.)
            img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                             fy=self.pre_scale_factor,
                             interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize=None, fx=self.pre_scale_factor,
                               fy=self.pre_scale_factor,
                               interpolation=cv2.INTER_NEAREST)

        if self.use_random_scale:
            # bilinear interp requires float img
            scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                           self.scale_min
            img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize=None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_NEAREST)

        # random crop to input sz
        img, coords = pad_and_or_crop(img, self.input_sz, mode="random")
        label, _ = pad_and_or_crop(label, self.input_sz, mode="fixed",
                                   coords=coords)

        _, mask_img1 = self._filter_label(label)
        # uint8 tensor as masks should be binary, also for consistency with
        # prepare_train, but converted to float32 in main loop because is used
        # multiplicatively in loss
        mask_img1 = torch.from_numpy(mask_img1.astype(np.uint8)).cuda()

        # converting to PIL does not change underlying np datatype it seems
        img1 = Image.fromarray(img.astype(np.uint8))

        img1 = self.jitter_tf(img1)  # not in place, new memory
        img1 = np.array(img1)

        # channels still last
        if not self.no_sobel:
            img1 = custom_greyscale_numpy(img1, include_rgb=self.include_rgb)

        img1 = img1.astype(np.float32) / 255.

        # convert both to channel-first tensor format
        # make them all cuda tensors now, except label, for optimality
        img1 = torch.from_numpy(img1).permute(2, 0, 1).cuda()

        # mask if required
        if self.mask_input:
            masked = 1 - mask_img1
            img1[:, masked] = 0

        if self.use_random_affine:
            affine_kwargs = {"min_rot": self.aff_min_rot, "max_rot": self.aff_max_rot,
                             "min_shear": self.aff_min_shear,
                             "max_shear": self.aff_max_shear,
                             "min_scale": self.aff_min_scale,
                             "max_scale": self.aff_max_scale}
            img1, _, _ = random_affine(img1, **affine_kwargs)  # tensors

        if np.random.rand() > self.flip_p:
            img1 = torch.flip(img1, dims=[2])  # horizontal, along width

        if RENDER_DATA:
            render(img1, mode="image", name=("train_data_img1_%d" % index))
            render(mask_img1, mode="mask", name=("train_data_mask_%d" % index))

        return img1, mask_img1

    def _prepare_test(self, index, img, label):
        print("#########################################################")
        # This returns cpu tensors.
        #   Image: 3D with channels last, float32, in range [0, 1] (normally done
        #     by ToTensor).
        #   Label map: 2D, flat int64, [0 ... sef.gt_k - 1]
        # label is passed in canonical [0 ... 181] indexing

        assert (img.shape[:2] == label.shape)
        img = img.astype(np.float32)
        label = label.astype(np.int32)

        # shrink original images, for memory purposes, otherwise no point
        if self.pre_scale_all:
            assert (self.pre_scale_factor < 1.)
            img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                             fy=self.pre_scale_factor,
                             interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize=None, fx=self.pre_scale_factor,
                               fy=self.pre_scale_factor,
                               interpolation=cv2.INTER_NEAREST)

        # center crop to input sz
        img, _ = pad_and_or_crop(img, self.input_sz, mode="centre")
        label, _ = pad_and_or_crop(label, self.input_sz, mode="centre")

        # finish
        if not self.no_sobel:
            img = custom_greyscale_numpy(img, include_rgb=self.include_rgb)

        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)

        if RENDER_DATA:
            render(label, mode="label", name=("test_data_label_pre_%d" % index))

        # convert to coarse if required, reindex to [0, gt_k -1], and get mask
        label, mask = self._filter_label(label)

        # mask if required
        if self.mask_input:
            masked = 1 - mask
            img[:, masked] = 0

        if RENDER_DATA:
            render(img, mode="image", name=("test_data_img_%d" % index))
            render(label, mode="label", name=("test_data_label_post_%d" % index))
            render(mask, mode="mask", name=("test_data_mask_%d" % index))
        # dataloader must return tensors (conversion forced in their code anyway)

        # trans = torchvision.transforms.ToPILImage()
        # outi = trans(img)
        # outi.show()
        # outi = trans(torch.from_numpy(label))
        # outi.show()

        return img, torch.from_numpy(label), torch.from_numpy(mask.astype(np.uint8))

    def __getitem__(self, index):
        image_id = self.files[index]
        image, label = self._load_data(image_id)
        # print("WE ARE IN GET ITEM")
        if self.purpose == "train":
            if not self.single_mode:  # self.single_mode: False
                return self._prepare_train(index, image, label)
            else:
                return self._prepare_train_single(index, image, label)
        else:
            assert (self.purpose == "test")
            return self._prepare_test(index, image, label)

    def __len__(self):
        return len(self.files)

    def _check_gt_k(self):
        raise NotImplementedError()

    def _filter_label(self):
        raise NotImplementedError()

    def _set_files(self):
        raise NotImplementedError()

    def _load_data(self, image_id):
        raise NotImplementedError()


class _Coco164kCuratedFew(_Coco):
    """Base class
  This contains fields and methods common to all COCO 164k curated few datasets:
  (curated) Coco164kFew_Stuff
  (curated) Coco164kFew_Stuff_People
  (curated) Coco164kFew_Stuff_Animals
  (curated) Coco164kFew_Stuff_People_Animals
  """

    def __init__(self, **kwargs):
        super(_Coco164kCuratedFew, self).__init__(**kwargs)

        # work out name
        config = kwargs["config"]
        assert (config.use_coarse_labels)  # we only deal with coarse labels
        self.include_things_labels = config.include_things_labels  # people
        self.incl_animal_things = config.incl_animal_things  # animals

        version = config.coco_164k_curated_version

        name = "Coco164kFew_Stuff"
        if self.include_things_labels and self.incl_animal_things:
            name += "_People_Animals"
        elif self.include_things_labels:
            name += "_People"
        elif self.incl_animal_things:
            name += "_Animals"
        self.name = (name + "_%d" % version)  # Coco164kFew_Stuff_6  (version=6)
        print("Specific type of _Coco164kCuratedFew dataset: %s" % self.name)

        self._set_files()

    def _set_files(self):
        # Create data list by parsing the "images" folder
        if self.split in ["train2017", "val2017"]:
            file_list = osp.join(self.root, "curated", self.split, self.name + ".txt")
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]

            self.files = file_list

        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, image_id):
        # Set paths
        image_path = osp.join(self.root, "images", self.split, image_id + ".jpg")
        label_path = osp.join(self.root, "annotations", self.split, image_id + ".png")
        # print("image_id: ", image_id)
        # print("image_path: ", image_path)
        # print("label_path: ", label_path)

        # image_path = osp.join(self.root, "images", "val2017", image_id + ".jpg").replace("\\","/")
        # label_path = osp.join(self.root, "annotations", "stuffthingmaps_trainval2017_2", "val2017",
        #                       image_id + ".png").replace("\\","/")

        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
        label[label == 255] = -1  # to be consistent with 10k

        return image, label


class _CocoFew(_Coco):
    """
  This contains methods for the following datasets
  COCO-few (6)
  COCOStuff-few (3)
  """

    def __init__(self, **kwargs):
        super(_CocoFew, self).__init__(**kwargs)

        config = kwargs["config"]
        assert (config.use_coarse_labels)  # we only deal with coarse labels
        self.include_things_labels = config.include_things_labels
        self.incl_animal_things = config.incl_animal_things

        self._check_gt_k()

        # indexes correspond to order in these lists
        self.label_names = [
            "sky-stuff",
            "plant-stuff",
            "ground-stuff",
        ]

        # CHANGED. Can have animals and/or people.
        if self.include_things_labels:
            self.label_names += ["person-things"]

        if self.incl_animal_things:
            self.label_names += ["animal-things"]

        assert (len(self.label_names) == self.gt_k)

        # make dict that maps fine labels to our labels
        self._fine_to_few_dict = self._make_fine_to_few_dict()

    def _make_fine_to_few_dict(self):
        # only make indices
        self.label_orig_coarse_inds = []
        for label_name in self.label_names:
            orig_coarse_ind = cocostuff_fine_to_coarse._sorted_coarse_names.index(
                label_name)
            self.label_orig_coarse_inds.append(orig_coarse_ind)  # [23, 22, 21]

        # excludes -1 (fine - see usage in filter label - as with Coco10kFull)
        _fine_to_few_dict = {}
        for c in range(182):
            orig_coarse_ind = self._fine_to_coarse_dict[c]  # defined in init og COCO! (0-26)

            if orig_coarse_ind in self.label_orig_coarse_inds:  # in [23, 22, 21]
                new_few_ind = self.label_orig_coarse_inds.index(orig_coarse_ind)  # 0,1,2
                print("assign fine %d coarse %d to new ind %d" % (c, orig_coarse_ind, new_few_ind))
            else:
                new_few_ind = -1
            _fine_to_few_dict[c] = new_few_ind

        return _fine_to_few_dict

    def _check_gt_k(self):
        # Can have animals and/or people.
        expected_gt_k = 3
        if self.include_things_labels:
            expected_gt_k += 1
        if self.incl_animal_things:
            expected_gt_k += 1

        assert (self.gt_k == expected_gt_k)

    def _filter_label(self, label):
        new_label_map = np.zeros(label.shape, dtype=label.dtype)
        for c in range(182):
            new_label_map[label == c] = self._fine_to_few_dict[c]

        mask = (new_label_map >= 0)

        assert (mask.dtype == np.bool)
        return new_label_map, mask


class Coco164kCuratedFew(_Coco164kCuratedFew, _CocoFew):
    def __init__(self, **kwargs):
        super(Coco164kCuratedFew, self).__init__(**kwargs)
