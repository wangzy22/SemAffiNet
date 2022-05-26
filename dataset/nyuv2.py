import math
import random
import torch
import numpy as np
from os.path import join
from glob import glob
import SharedArray as SA
import imageio
import multiprocessing as mp
from os.path import join, exists
import os
import cv2
import open3d as o3d
from copy import deepcopy

import torch.utils.data as data
import dataset.augmentation_2d_link as t_2d
import dataset.augmentation as t
from dataset.voxelizer import Voxelizer
from dataset.scanNet3D import sa_create


# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


class LinkCreator(object):
    def __init__(self, image_dim=(640, 480), voxelSize=0.05):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        self.intrinsic = intrinsic.intrinsic_matrix
        self.imageDim = image_dim
        self.voxel_size = voxelSize

    def computeLinking(self, coords, depth):
        """
        :param coords: N x 3 format
        :param depth: H x W format
        :return: linking, N x 3 format, (H,W,mask)
        """
        link = np.zeros((3, coords.shape[0]), dtype=np.int)
        p = coords.T

        p[0] = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        p[1] = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        pi = np.round(p).astype(np.int)
        pi[0] = -pi[0] + self.imageDim[1] - 1
        inside_mask = (pi[0] >= 0) * (pi[1] >= 0) \
                      * (pi[0] < self.imageDim[1]) * (pi[1] < self.imageDim[0])
        link[0][inside_mask] = pi[1][inside_mask]
        link[1][inside_mask] = pi[0][inside_mask]
        link[2][inside_mask] = 1

        return link.T


class NYUv2Dataset(data.Dataset):
    VIEW_NUM = 1
    IMG_DIM = (480, 640)
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, dataPathPrefix='Data', voxelSize=0.05,
                 split='train', aug=False, memCacheInit=False,
                 identifier=1296, loop=1,
                 data_aug_color_trans_ratio=0.1,
                 data_aug_color_jitter_std=0.05, data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2, eval_all=False,
                 classes=13
                 ):
        super(NYUv2Dataset, self).__init__()
        self.split = split
        self.classes = classes
        self.identifier = identifier
        datalist_fp = join(dataPathPrefix, split+'.txt')
        self.data_paths = self.read_imglist(datalist_fp)
        self.voxelSize = voxelSize
        self.aug = aug
        self.loop = loop
        self.eval_all = eval_all
        self.root = dataPathPrefix

        self.voxelizer = Voxelizer(
            voxel_size=voxelSize,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        if aug:
            prevoxel_transform_train = [t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(data_aug_hue_max, data_aug_saturation_max),
            ]
            self.input_transforms = t.Compose(input_transforms)

        self.linkCreator = LinkCreator(image_dim=self.IMG_DIM, voxelSize=voxelSize)
        # 2D AUG
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        if self.aug:
            self.transform_2d = t_2d.Compose([
                t_2d.RandomHorizontalFlip(),
                t_2d.RandomVerticalFlip(),
                t_2d.RandomGaussianBlur(),
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])
        else:
            self.transform_2d = t_2d.Compose([
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.data_paths)

    def read_imglist(self, imglist_fp):
        ll = []
        with open(imglist_fp, 'r') as fd:
            for line in fd:
                ll.append(line.strip())
        return ll

    def read_inpus(self, imgname):
        img_fp = os.path.join(self.root, 'image', imgname + '.png')
        img = cv2.imread(img_fp).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth_fp = os.path.join(self.root, 'depth', imgname + '.npy')
        depth = np.load(depth_fp)

        pc_fp = os.path.join(self.root, 'pc_o3d', imgname + '.npy')
        pc = np.load(pc_fp)

        mask_fp = os.path.join(self.root, 'label' + str(self.classes), imgname + '.png')
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)
        mask -= 1       # 0->255

        return img, depth, pc, mask

    def __getitem__(self, index_long):
        index = index_long % len(self.data_paths)
        # name_2d = self.data_paths[index]
        colors, depth, pc, labels_2d = self.read_inpus(self.data_paths[index])
        locs_in = pc[:, :3]
        feats_in = pc[:, 3:6] * 255
        labels_in = pc[:, 6] - 1
        labels_in[labels_in < 0] = 255

        links = self.get_link(deepcopy(locs_in), deepcopy(depth))

        r = np.arange(locs_in.shape[0])
        np.random.shuffle(r)
        locs_in = locs_in[r]
        feats_in = feats_in[r]
        labels_in = labels_in[r]
        links = links[r]

        locs_in[:, 0] = locs_in[:, 0] - locs_in[:, 0].min()
        locs_in[:, 1] = locs_in[:, 1] - locs_in[:, 1].min()
        locs_in[:, 2] = locs_in[:, 2] - locs_in[:, 2].min()
        locs_in = locs_in * 10

        colors, labels_2d, links = self.transform_2d(colors, labels_2d, links)
        colors, labels_2d = torch.unsqueeze(colors, dim=-1), torch.unsqueeze(labels_2d, dim=-1)

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
        locs = locs_in
        locs, feats, labels, inds_reconstruct, links = self.voxelizer.voxelize(locs, feats_in, labels_in, link=links)
        self.check_link(links, labels_2d, labels)
        
        if self.eval_all:
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        feats = torch.from_numpy(feats).float() / 127.5 - 1
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return coords, feats, labels, colors, labels_2d, links, torch.from_numpy(inds_reconstruct).long()
        return coords, feats, labels, colors, labels_2d, links

    def get_link(self, coords, depth):
        """
        :param      room_id:
        :param      coords: Nx3
        :return:    imgs:   CxHxWxV Tensor
                    labels: HxWxV Tensor
                    links: Nx4xV(1,H,W,mask) Tensor
        """
        link = np.ones([coords.shape[0], 4], dtype=np.int)
        link[:, 1:4] = self.linkCreator.computeLinking(coords, depth)

        return link

    def check_link(self, links, labels_2d, labels_3d):
        if len((labels_3d != 255).nonzero()[0]) > 0:
            for idx in (labels_3d != 255).nonzero()[0]:
                l3d = labels_3d[idx]
                link = links[idx, :, 0]
                l2d = labels_2d[link[1], link[2]]
                assert int(l3d) == int(l2d), idx


def collation_fn(batch):
    """
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
    """
    coords, feats, labels, colors, labels_2d, links = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i
        links[i][:, 0, :] *= i

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
           torch.stack(colors), torch.stack(labels_2d), torch.cat(links)


def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON
    """
    coords, feats, labels, colors, labels_2d, links, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        links[i][:, 0, :] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
           torch.stack(colors), torch.stack(labels_2d), torch.cat(links), torch.cat(inds_recons)