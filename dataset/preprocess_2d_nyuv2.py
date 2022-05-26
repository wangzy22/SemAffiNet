import os
import h5py
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from PIL import Image
import open3d as o3d
import torch


def getPointCloudFromRGBD(rgb, depth, label):
    depth = np.expand_dims(depth, axis=-1)
    label = np.expand_dims(label, axis=-1).astype(np.uint16)
    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)
    label = o3d.geometry.Image(label)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
    ld = o3d.geometry.RGBDImage.create_from_color_and_depth(label, depth)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pc_label = o3d.geometry.PointCloud.create_from_rgbd_image(ld, intrinsic)
    pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pc_label.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    assert np.array_equal(np.asarray(pc.points), np.asarray(pc_label.points)) 
    pc_out = np.concatenate([np.asarray(pc.points), np.asarray(pc.colors), np.asarray(pc_label.colors)[:, 0:1]], axis=-1)
    return pc_out


def get_nyu_cam_mats(path_nyu_cam_mats):
    mats = []
    with open(path_nyu_cam_mats, 'r') as f_cam:
        lines = f_cam.readlines()
        for line_id in range(0, len(lines), 4):
            mat = np.zeros([3, 3], dtype=np.float)
            for i in range(3):
                line = lines[line_id + i]
                eles = line.split(' ')
                for j in range(len(eles)):
                    if i == 2:
                        mat[i][j] = float(eles[j])
                    else:
                        mat[i][j] = float(eles[j]) * 1000
            mats.append(mat)
    return mats


def read_label_map(path_map):
    f_map = scio.loadmat(path_map)
    if 'classMapping13' in f_map.keys():
        map_class = np.array(f_map['classMapping13'][0][0][0])[0]
    else:
        map_class = f_map['mapClass'][0]
    print(map_class.shape)
    dict_map = {0: 0}
    for ori_id, mapped_id in enumerate(map_class):
        dict_map[ori_id + 1] = mapped_id
    return dict_map


def save_imgs(f_h5, dir_img_out):
    if not os.path.isdir(dir_img_out):
        os.makedirs(dir_img_out)

    images = np.array(f_h5["images"])
    bar = tqdm(enumerate(images))
    for i, a in bar:
        r = Image.fromarray(a[0]).convert('L')
        g = Image.fromarray(a[1]).convert('L')
        b = Image.fromarray(a[2]).convert('L')
        img = Image.merge("RGB", (r, g, b))
        img = img.transpose(Image.ROTATE_270)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_path = os.path.join(dir_img_out, "%06d.png" % (i+1))
        img.save(img_path, optimize=True)

        bar.set_description('rgb {}'.format(str(i)))


def save_depths(f_h5, dir_img_out):
    if not os.path.isdir(dir_img_out):
        os.makedirs(dir_img_out)

    depths = np.array(f_h5["depths"])
    bar = tqdm(enumerate(depths))
    for i, depth in bar:
        depth = depth.transpose((1, 0))
        depth_path = os.path.join(dir_img_out, "%06d.npy" % (i+1))
        np.save(depth_path, depth)

        bar.set_description('depth {}'.format(str(i)))


def save_pc_o3d(f_h5, maps, dir_pc_out):
    if not os.path.isdir(dir_pc_out):
        os.makedirs(dir_pc_out)

    depths = np.array(f_h5["depths"])
    images = np.array(f_h5["images"])
    labels = np.array(f_h5["labels"])
    bar = tqdm(enumerate(zip(images, depths, labels)))
    for i, (image, depth, label) in bar:
        if maps:
            for map in maps:
                label = np.vectorize(map.get)(label)
        image = torch.from_numpy(image).permute(2, 1, 0).contiguous().numpy()
        depth = torch.from_numpy(depth).transpose(1, 0).contiguous().numpy()
        label = torch.from_numpy(label).transpose(1, 0).contiguous().numpy()
        pc = getPointCloudFromRGBD(image, depth*100, label)
        pc_path = os.path.join(dir_pc_out, "%06d.npy" % (i+1))
        np.save(pc_path, pc)

        bar.set_description('pc {}'.format(str(i)))


def save_labels(f_h5, maps, dir_label_out):
    if not os.path.isdir(dir_label_out):
        os.makedirs(dir_label_out)

    labels = np.array(f_h5["labels"])
    bar = tqdm(enumerate(labels))
    for i, label in bar:
        if maps:
            for map in maps:
                label = np.vectorize(map.get)(label)
        label = label.transpose((1, 0))
        label_img = Image.fromarray(np.uint8(label))
        label_path = os.path.join(dir_label_out, "%06d.png" % (i+1))
        label_img.save(label_path, 'PNG', optimize=True)

        bar.set_description('label {}'.format(str(i)))

def save_list(pth_splits, pth_train, pth_test):
    def write_txt(f_list, list_ids):
        f_list.write('\n'.join(list_ids))
        f_list.close()

    train_test = scio.loadmat(pth_splits)
    train_images = tuple([int(x) for x in train_test["trainNdxs"]])
    test_images = tuple([int(x) for x in train_test["testNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    train_ids = ["%06d" % i for i in train_images]
    test_ids = ["%06d" % i for i in test_images]

    train_list_file = open(pth_train, 'w')
    write_txt(train_list_file, train_ids)

    test_list_file = open(pth_test, 'w')
    write_txt(test_list_file, test_ids)


def main(dir_meta, dir_out):
    path_nyu_depth_v2_labeled = os.path.join(dir_meta, "nyu_depth_v2_labeled.mat")
    f = h5py.File(path_nyu_depth_v2_labeled)
    print(f.keys())

    dir_sub_out = os.path.join(dir_out, 'image')
    save_imgs(f, dir_sub_out)

    dir_sub_out = os.path.join(dir_out, 'depth')
    save_depths(f, dir_sub_out)

    pth_map_label40 = os.path.join(dir_meta, "classMapping40.mat")
    pth_map_label13 = os.path.join(dir_meta, "class13Mapping.mat")
    dir_sub_out = os.path.join(dir_out, 'pc_o3d')
    save_pc_o3d(f, [read_label_map(pth_map_label40), read_label_map(pth_map_label13)], dir_sub_out)

    pth_map_label13 = os.path.join(dir_meta, "class13Mapping.mat")
    dir_sub_out = os.path.join(dir_out, 'label13')
    save_labels(f, [read_label_map(pth_map_label40), read_label_map(pth_map_label13)], dir_sub_out)

    pth_splits = os.path.join(dir_meta, "splits.mat")
    pth_train = os.path.join(dir_out, 'train.txt')
    pth_test = os.path.join(dir_out, 'test.txt')
    save_list(pth_splits, pth_train, pth_test)


if __name__ == '__main__':
    input_dir = "data/NYUv2/original"
    output_dir = "data/NYUv2"

    main(input_dir, output_dir)