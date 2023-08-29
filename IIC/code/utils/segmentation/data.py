import sys
from datetime import datetime
import torch
from torch.utils.data import ConcatDataset
from IIC.code.datasets.segmentation import cocostuff
# from IIC.code.datasets.segmentation import potsdam
# from IIC.code.datasets.segmentation import DoerschDataset


def segmentation_create_dataloaders(config):  # define partitions and call dataloader
    if config.mode == "IID":
        if "Coco10k" in config.dataset:
            config.train_partitions = ["all"]
            config.mapping_assignment_partitions = ["all"]
            config.mapping_test_partitions = ["all"]
        elif "Coco164k" in config.dataset:
            config.train_partitions = ["train2017", "val2017"] #for_colab
            config.mapping_assignment_partitions = ["train2017", "val2017"]
            config.mapping_test_partitions = ["train2017", "val2017"]
            # config.train_partitions = [ "val2017"]
            # config.mapping_assignment_partitions = [ "val2017"]
            # config.mapping_test_partitions = [ "val2017"]
        else:
            raise NotImplementedError

    if "Coco" in config.dataset:
        dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
            make_Coco_dataloaders(config)
    else:
        raise NotImplementedError

    return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader


def make_Coco_dataloaders(config):
    print("WE ARE IN make_Coco_dataloaders!")
    dataset_class = cocostuff.__dict__[config.dataset]
    # print("dataset_class: ", dataset_class)# <class 'IIC.code.datasets.segmentation.cocostuff.Coco164kCuratedFew'>
    # print("cocostuff: ", cocostuff)  # module 'IIC.code.datasets.segmentation.cocostuff'
    # print("config.dataset: ", config.dataset)  # Coco164kCuratedFew

    dataloaders = _create_dataloaders(config, dataset_class)  # it has partitions too!

    mapping_assignment_dataloader = \
        _create_mapping_loader(config, dataset_class,
                               partitions=config.mapping_assignment_partitions)

    mapping_test_dataloader = \
        _create_mapping_loader(config, dataset_class,
                               partitions=config.mapping_test_partitions)

    return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader


def _create_dataloaders(config, dataset_class):
    print("WE ARE IN _create_dataloaders")

    # unlike in clustering, each dataloader here returns pairs of images - we
    # need the matrix relation between them
    dataloaders = []
    do_shuffle = (config.num_dataloaders == 1)  # TRUE
    for d_i in range(config.num_dataloaders):  # 1 dl for now
        print("Creating paired dataloader %d out of %d time %s" %
              (d_i, config.num_dataloaders, datetime.now()))
        sys.stdout.flush()

        train_imgs_list = []
        for train_partition in config.train_partitions:
            train_imgs_curr = dataset_class(
                **{"config": config,
                   "split": train_partition,
                   "purpose": "train"}  # return training tuples, not including labels
            )
            # if config.use_doersch_datasets:
            #   train_imgs_curr = DoerschDataset(config, train_imgs_curr)

            train_imgs_list.append(train_imgs_curr)

        train_imgs = ConcatDataset(train_imgs_list)

        train_dataloader = torch.utils.data.DataLoader(train_imgs,
                                                       batch_size=config.dataloader_batch_sz,
                                                       shuffle=do_shuffle,
                                                       num_workers=0,
                                                       drop_last=False)

        if d_i > 0:
            assert (len(train_dataloader) == len(dataloaders[d_i - 1]))

        dataloaders.append(train_dataloader)

    num_train_batches = len(dataloaders[0])
    print("Length of paired datasets vector %d" % len(dataloaders))
    print("Number of batches per epoch: %d" % num_train_batches)
    sys.stdout.flush()

    return dataloaders  # list[DataLoader]


def _create_mapping_loader(config, dataset_class, partitions):
    imgs_list = []  # list of classes
    for partition in partitions:  # ["val2017"]
        imgs_curr = dataset_class(
            **{"config": config,
               "split": partition,
               "purpose": "test"}  # return testing tuples, image and label

        )  # <IIC.code.datasets.segmentation.cocostuff.Coco164kCuratedFew object>
        imgs_list.append(imgs_curr)

    imgs = ConcatDataset(imgs_list)  # <class 'torch.utils.data.dataset.ConcatDataset'>
    dataloader = torch.utils.data.DataLoader(imgs,
                                             batch_size=config.batch_sz,
                                             # full batch
                                             shuffle=False,
                                             # no point since not trained on
                                             num_workers=0,
                                             drop_last=False)
    return dataloader  # DataLoader
