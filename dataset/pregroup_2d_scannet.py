import os
import torch
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm

from pointnet2_ops import pointnet2_utils
from dataset.scanNetCross import LinkCreator


def main(root, split):
    linkCreator = LinkCreator(image_dim=(320, 240), voxelSize=0.05)
    data3d_paths = sorted(glob(os.path.join(root, '3D', split, '*.pth')))
    data2d_paths = []
    room_groups = {}
    for x in data3d_paths:
        ps = glob(os.path.join(x[:-15].replace(split, '2D'), 'color', '*.jpg'))
        ps.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        data2d_paths.append(ps)
    for room_id in tqdm(range(len(data3d_paths))):
        coords, _, _ = torch.load(data3d_paths[room_id])
        quartile = (coords[:, -1].max() - coords[:, -1].min()) * 0.1
        qlower = coords[:, -1].min() + quartile
        qupper = coords[:, -1].max() - quartile
        coords_room = torch.from_numpy(coords[(coords[:, -1] > qlower) & (coords[:, -1] < qupper)]).unsqueeze(dim=0).cuda()
        fps_idx = pointnet2_utils.furthest_point_sample(coords_room, 5)
        fps_coords = pointnet2_utils.gather_operation(coords_room.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous().cpu()

        frames_path = data2d_paths[room_id]
        range_frames = []
        centroid_frames = []
        delete_frames = []
        for f in frames_path:
            depth = imageio.imread(f.replace('color', 'depth').replace('jpg', 'png')) / 1000.0
            posePath = f.replace('color', 'pose').replace('.jpg', '.txt')
            pose = np.asarray(
                [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                 (x.split(" ") for x in open(posePath).read().splitlines())]
            )
            H, W = depth.shape
            link = linkCreator.computeLinking(pose, coords, depth)
            coords_link = coords[link[:, 2] == 1]
            if len(coords_link) < 10:
                delete_frames.append(f.split('/')[-1])
                continue
            mu = coords_link.mean(axis=0, keepdims=True)
            sigma = coords_link.std(axis=0, keepdims=True)
            coords_link = coords_link[(coords_link < mu + 3 * sigma).all(axis=1) & (coords_link > mu - 3 * sigma).all(axis=1)]
            centroid = coords_link.mean(axis=0)
            centroid_frames.append(centroid)
            range_min = coords_link.min(axis=0)
            range_max = coords_link.max(axis=0)
            range_xyz = np.stack([range_min, range_max], axis=0)
            range_frames.append(range_xyz)
        range_frames = torch.from_numpy(np.stack(range_frames, axis=0))
        centroid_frames = torch.from_numpy(np.stack(centroid_frames, axis=0))
        distance = torch.norm(fps_coords - centroid_frames.unsqueeze(dim=1), p=2, dim=-1)
        group_nearest = distance.argmin(dim=1)
        room_name = data3d_paths[room_id].split('/')[2][:12]
        room_groups[room_name] = {}
        room_groups[room_name]['group_centroid'] = fps_coords
        room_groups[room_name]['frames_group'] = group_nearest
        room_groups[room_name]['frames_range'] = range_frames
        room_groups[room_name]['frames_centroid'] = centroid_frames
        room_groups[room_name]['frames_delete'] = delete_frames
    torch.save(room_groups, 'data/view_groups/view_groups_'+split+'pth')

def get_stats(group):
    return {'coords_z': group.sum()}


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    root = 'data'
    split = 'val'
    main(root, split)