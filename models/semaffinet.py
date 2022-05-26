#!/usr/bin/env python
from collections import OrderedDict


import torch
from torch import nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.unet_2d import ResUnet as model2D
from models.unet_3d import mink_unet as model3D
from models.bpm import Linking
from models.transformer_utils.transformer_predictor import TransformerPredictor, MLP


def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:]  # remove 'module.' of dataparallel
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def constructor3d(**kwargs):
    model = model3D(**kwargs)
    # model = model.cuda()
    return model


def constructor2d(**kwargs):
    model = model2D(**kwargs)
    # model = model.cuda()
    return model


class SemAffiNet(nn.Module):

    def __init__(self, cfg=None):
        super(SemAffiNet, self).__init__()
        self.viewNum = cfg.viewNum
        self.tr_dim = cfg.hidden_dim
        self.mask_dim = cfg.mask_dim
        # 2D
        net2d = constructor2d(layers=cfg.layers_2d, classes=cfg.classes, out_channels=self.mask_dim)
        self.layer0_2d = net2d.layer0
        self.layer1_2d = net2d.layer1
        self.layer2_2d = net2d.layer2
        self.layer3_2d = net2d.layer3
        self.layer4_2d = net2d.layer4

        self.class_convfeat_2d = net2d.class_convfeat
        self.relu_2d = net2d.relu

        self.up4_2d = net2d.up4
        self.class_conv4_2d = net2d.class_conv4
        self.bn4_2d = net2d.bn4
        self.sat_w4_2d = MLP(self.tr_dim, self.tr_dim, self.mask_dim, 5)
        self.sat_b4_2d = MLP(self.tr_dim, self.tr_dim, self.mask_dim, 5)
        self.delayer4_2d = net2d.delayer4

        self.up3_2d = net2d.up3
        self.class_conv3_2d = net2d.class_conv3
        self.bn3_2d = net2d.bn3
        self.sat_w3_2d = MLP(self.tr_dim, self.tr_dim, self.mask_dim, 5)
        self.sat_b3_2d = MLP(self.tr_dim, self.tr_dim, self.mask_dim, 5)
        self.delayer3_2d = net2d.delayer3

        self.up2_2d = net2d.up2
        self.class_conv2_2d = net2d.class_conv2
        self.bn2_2d = net2d.bn2
        self.sat_w2_2d = MLP(self.tr_dim, self.tr_dim, self.mask_dim, 5)
        self.sat_b2_2d = MLP(self.tr_dim, self.tr_dim, self.mask_dim, 5)
        self.delayer2_2d = net2d.delayer2

        self.cls_2d = net2d.cls
        self.num_layers_2d = cfg.layers_2d

        if cfg.layers_2d >= 50:
            self.down_layer1_2d = nn.Conv2d(256, 64, 1)
            self.down_layer2_2d = nn.Conv2d(512, 128, 1)
            self.down_layer3_2d = nn.Conv2d(1024, 256, 1)
            self.down_layer4_2d = nn.Conv2d(2048, 512, 1)

        # 3D
        net3d = constructor3d(in_channels=3, out_channels=cfg.mask_dim, D=3, arch=cfg.arch_3d)
        self.layer0_3d = nn.Sequential(net3d.conv0p1s1, net3d.bn0, net3d.relu)
        self.layer1_3d = nn.Sequential(net3d.conv1p1s2, net3d.bn1, net3d.relu, net3d.block1)
        self.layer2_3d = nn.Sequential(net3d.conv2p2s2, net3d.bn2, net3d.relu, net3d.block2)
        self.layer3_3d = nn.Sequential(net3d.conv3p4s2, net3d.bn3, net3d.relu, net3d.block3)
        self.layer4_3d = nn.Sequential(net3d.conv4p8s2, net3d.bn4, net3d.relu, net3d.block4)

        self.class_convfeat = net3d.class_convfeat
        self.relu_3d = net3d.relu
        out_channels = net3d.out_channels

        self.convtr4p16s2 = net3d.convtr4p16s2
        self.class_conv4 = net3d.class_conv4
        self.bntr4 = net3d.bntr4
        self.sat_w4 = MLP(self.tr_dim, self.tr_dim, out_channels, 5)
        self.sat_b4 = MLP(self.tr_dim, self.tr_dim, out_channels, 5)
        self.block5 = net3d.block5

        self.convtr5p8s2 = net3d.convtr5p8s2
        self.class_conv5 = net3d.class_conv5
        self.bntr5 = net3d.bntr5
        self.sat_w5 = MLP(self.tr_dim, self.tr_dim, out_channels, 5)
        self.sat_b5 = MLP(self.tr_dim, self.tr_dim, out_channels, 5)
        self.block6 = net3d.block6

        self.convtr6p4s2 = net3d.convtr6p4s2
        self.class_conv6 = net3d.class_conv6
        self.bntr6 = net3d.bntr6
        self.sat_w6 = MLP(self.tr_dim, self.tr_dim, out_channels, 5)
        self.sat_b6 = MLP(self.tr_dim, self.tr_dim, out_channels, 5)
        self.block7 = net3d.block7
        
        self.convtr7p2s2 = net3d.convtr7p2s2
        self.bntr7 = net3d.bntr7
        self.block8 = net3d.block8
        self.cls_3d = net3d.final

        self.predictor = TransformerPredictor(config=cfg, in_channels_3d=net3d.PLANES[3], in_channels_2d=512,
                                              num_classes=cfg.classes, D=3)

        # Linker
        self.linker_p2 = Linking(96, net3d.PLANES[6], viewNum=self.viewNum)
        self.linker_p3 = Linking(128, net3d.PLANES[5], viewNum=self.viewNum)
        self.linker_p4 = Linking(256, net3d.PLANES[4], viewNum=self.viewNum)
        self.linker_p5 = Linking(512, net3d.PLANES[3], viewNum=self.viewNum)        

    def SAT(self, pred_class, coordinates, query_weight, query_bias):
        pred_class = torch.sigmoid(pred_class)
        query_weight = torch.abs(query_weight)
        weight_list = []
        bias_list = []
        for b in coordinates[:, 0].unique():
            pred_class_b = pred_class[coordinates[:, 0] == b]
            weight = torch.einsum("nm,mc->nc", pred_class_b, query_weight[b])
            bias = torch.einsum("nm,mc->nc", pred_class_b, query_bias[b])
            weight_list.append(weight)
            bias_list.append(bias)

        return torch.cat(weight_list, dim=0), torch.cat(bias_list, dim=0)

    def mul_mask_voxel(self, mask_embed, voxel_embed):
        outputs_seg_lvl = []
        outputs_coords_lvl = []
        for i in range(len(voxel_embed.C[:, 0].unique())):
            idx_i = voxel_embed.C[:, 0] == i
            seg_mask = torch.einsum("qc,cn->qn", mask_embed[i], voxel_embed.F[idx_i].transpose(0, 1).contiguous())
            outputs_seg_lvl.append(seg_mask.transpose(0, 1).contiguous())
            outputs_coords_lvl.append(voxel_embed.C[idx_i])
        outputs_seg_lvl = torch.cat(outputs_seg_lvl, dim=0)
        outputs_coords_lvl = torch.cat(outputs_coords_lvl, dim=0)
        assert torch.equal(voxel_embed.C, outputs_coords_lvl)
        return outputs_seg_lvl

    def forward(self, sparse_3d, images, links, targets_mid_3d=None):
        """
        images:BCHWV
        """
        ########################
        ##### Encoder Part #####
        ########################

        ### 2D feature extract
        x_size = images.size()
        h, w = x_size[2], x_size[3]
        data_2d = images.permute(4, 0, 1, 2, 3).contiguous()  # VBCHW
        data_2d = data_2d.view(x_size[0] * x_size[4], x_size[1], x_size[2], x_size[3])
        x = self.layer0_2d(data_2d)  # 1/4
        x2 = self.layer1_2d(x)  # 1/4
        x3 = self.layer2_2d(x2)  # 1/8
        x4 = self.layer3_2d(x3)  # 1/16
        x5 = self.layer4_2d(x4)  # 1/32

        if self.num_layers_2d >= 50:
            x2 = self.down_layer1_2d(x2)
            x3 = self.down_layer2_2d(x3)
            x4 = self.down_layer3_2d(x4)
            x5 = self.down_layer4_2d(x5)

        ### 3D feature extract
        out_p1 = self.layer0_3d(sparse_3d)
        out_b1p2 = self.layer1_3d(out_p1)
        out_b2p4 = self.layer2_3d(out_b1p2)
        out_b3p8 = self.layer3_3d(out_b2p4)
        out_b4p16 = self.layer4_3d(out_b3p8)  # corresponding to FPN p5

        mask_embed_2d, mask_embed, hs_sat_2d, hs_sat = self.predictor(x5, out_b4p16)
        predictions, predictions_2d = {}, {}
        b, n_c, d = hs_sat[-1].shape
        b2, n_p, d = hs_sat_2d[-1].shape
        outputs_seg_list = []
        outputs_seg_list_2d = []

        ########################
        ##### Decoder Part #####
        ########################

        ### Class prediction @ p5

        p5 = self.class_convfeat_2d(x5)
        seg_p5_2d = torch.einsum('bqc,bchw->bqhw', mask_embed_2d, p5)
        outputs_seg_list_2d.append(seg_p5_2d)

        outfeat = self.class_convfeat(out_b4p16)
        seg_feat_3d = self.mul_mask_voxel(mask_embed, outfeat)
        outputs_seg_list.append(seg_feat_3d)

        ### Linking @ p5
        V_B, C, H, W = x5.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p5, fused_2d_p5 = self.linker_p5(x5, out_b4p16, links_current_level, init_3d_data=sparse_3d)

        ### p5->p4, Class prediction @ p4, AdaBN @ p4, Block @ p4

        p4 = self.up4_2d(F.interpolate(fused_2d_p5, x4.shape[-2:], mode='bilinear', align_corners=True))
        p4 = self.class_conv4_2d(p4)
        seg_p4_2d = torch.einsum('bqc,bchw->bqhw', mask_embed_2d, p4)
        outputs_seg_list_2d.append(seg_p4_2d)

        p4_inst = self.bn4_2d(p4)
        hs_sat_w4_2d = self.sat_w4_2d(hs_sat_2d[-4].view(b2*n_p, d)).view(b2, n_p, -1)
        hs_sat_b4_2d = self.sat_b4_2d(hs_sat_2d[-4].view(b2*n_p, d)).view(b2, n_p, -1)
        weight4_2d = torch.einsum('bcd,bchw->bdhw', torch.abs(hs_sat_w4_2d), torch.sigmoid(seg_p4_2d))
        bias4_2d = torch.einsum('bcd,bchw->bdhw', hs_sat_b4_2d, torch.sigmoid(seg_p4_2d))
        p4_instnorm = p4_inst * weight4_2d + bias4_2d
        feat_p4 = self.relu_2d(p4_instnorm)
        p4 = torch.cat([feat_p4, x4], dim=1)
        p4 = self.delayer4_2d(p4)

        out4 = self.convtr4p16s2(fused_3d_p5)
        out4 = self.class_conv4(out4)
        seg_out4_3d = self.mul_mask_voxel(mask_embed, out4)
        outputs_seg_list.append(seg_out4_3d)

        out4_inst = self.bntr4(out4)
        hs_sat_w4 = self.sat_w4(hs_sat[-4].view(b*n_c, -1)).view(b, n_c, -1)
        hs_sat_b4 = self.sat_b4(hs_sat[-4].view(b*n_c, -1)).view(b, n_c, -1)
        weight, bias = self.SAT(seg_out4_3d, out4.C, hs_sat_w4, hs_sat_b4)
        out4_instnorm = out4_inst.F * weight + bias
        out4 = ME.SparseTensor(features=out4_instnorm, coordinate_map_key=out4.coordinate_map_key, coordinate_manager=out4.coordinate_manager)
        feat_out4 = self.relu_3d(out4)
        out_cat4 = ME.cat(feat_out4, out_b3p8)
        out4 = self.block5(out_cat4)

        ### Linking @ p4
        V_B, C, H, W = p4.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p4, fused_2d_p4 = self.linker_p4(p4, out4, links_current_level, init_3d_data=sparse_3d)

        ### p4->p3, Class prediction @ p3, AdaBN @ p3, Block @ p3
        p3 = self.up3_2d(F.interpolate(fused_2d_p4, x3.shape[-2:], mode='bilinear', align_corners=True))
        p3 = self.class_conv3_2d(p3)
        seg_p3_2d = torch.einsum('bqc,bchw->bqhw', mask_embed_2d, p3)
        outputs_seg_list_2d.append(seg_p3_2d)

        p3_inst = self.bn3_2d(p3)
        hs_sat_w3_2d = self.sat_w3_2d(hs_sat_2d[-3].view(b2*n_p, d)).view(b2, n_p, -1)
        hs_sat_b3_2d = self.sat_b3_2d(hs_sat_2d[-3].view(b2*n_p, d)).view(b2, n_p, -1)
        weight3_2d = torch.einsum('bcd,bchw->bdhw', torch.abs(hs_sat_w3_2d), torch.sigmoid(seg_p3_2d))
        bias3_2d = torch.einsum('bcd,bchw->bdhw', hs_sat_b3_2d, torch.sigmoid(seg_p3_2d))
        p3_instnorm = p3_inst * weight3_2d + bias3_2d
        feat_p3 = self.relu_2d(p3_instnorm)
        p3 = torch.cat([feat_p3, x3], dim=1)
        p3 = self.delayer3_2d(p3)

        out5 = self.convtr5p8s2(fused_3d_p4)
        out5 = self.class_conv5(out5)
        seg_out5_3d = self.mul_mask_voxel(mask_embed, out5)
        outputs_seg_list.append(seg_out5_3d)

        out5_inst = self.bntr5(out5)
        hs_sat_w5 = self.sat_w5(hs_sat[-3].view(b*n_c, -1)).view(b, n_c, -1)
        hs_sat_b5 = self.sat_b5(hs_sat[-3].view(b*n_c, -1)).view(b, n_c, -1)
        weight, bias = self.SAT(seg_out5_3d, out5.C, hs_sat_w5, hs_sat_b5)
        out5_instnorm = out5_inst.F * weight + bias
        out5 = ME.SparseTensor(features=out5_instnorm, coordinate_map_key=out5.coordinate_map_key, coordinate_manager=out5.coordinate_manager)
        feat_out5 = self.relu_3d(out5)
        out_cat5 = ME.cat(feat_out5, out_b2p4)
        out5 = self.block6(out_cat5)

        ### Linking @ p3
        V_B, C, H, W = p3.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p3, fused_2d_p3 = self.linker_p3(p3, out5, links_current_level, init_3d_data=sparse_3d)

        ### p3->p2, Class prediction @ p2, AdaBN @ p2, Block @ p2
        p2 = self.up2_2d(F.interpolate(fused_2d_p3, x2.shape[-2:], mode='bilinear', align_corners=True))
        p2 = self.class_conv2_2d(p2)
        seg_p2_2d = torch.einsum('bqc,bchw->bqhw', mask_embed_2d, p2)
        outputs_seg_list_2d.append(seg_p2_2d)

        p2_inst = self.bn2_2d(p2)
        hs_sat_w2_2d = self.sat_w2_2d(hs_sat_2d[-2].view(b2*n_p, d)).view(b2, n_p, -1)
        hs_sat_b2_2d = self.sat_b2_2d(hs_sat_2d[-2].view(b2*n_p, d)).view(b2, n_p, -1)
        weight2_2d = torch.einsum('bcd,bchw->bdhw', torch.abs(hs_sat_w2_2d), torch.sigmoid(seg_p2_2d))
        bias2_2d = torch.einsum('bcd,bchw->bdhw', hs_sat_b2_2d, torch.sigmoid(seg_p2_2d))
        p2_instnorm = p2_inst * weight2_2d + bias2_2d
        feat_p2 = self.relu_2d(p2_instnorm)
        p2 = torch.cat([feat_p2, x2], dim=1)
        p2 = self.delayer2_2d(p2)

        out6 = self.convtr6p4s2(fused_3d_p3)
        out6 = self.class_conv6(out6)
        seg_out6_3d = self.mul_mask_voxel(mask_embed, out6)
        outputs_seg_list.append(seg_out6_3d)

        out6_inst = self.bntr6(out6)
        hs_sat_w6 = self.sat_w6(hs_sat[-2].view(b*n_c, -1)).view(b, n_c, -1)
        hs_sat_b6 = self.sat_b6(hs_sat[-2].view(b*n_c, -1)).view(b, n_c, -1)
        weight, bias = self.SAT(seg_out6_3d, out6.C, hs_sat_w6, hs_sat_b6)
        out6_instnorm = out6_inst.F * weight + bias
        out6 = ME.SparseTensor(features=out6_instnorm, coordinate_map_key=out6.coordinate_map_key, coordinate_manager=out6.coordinate_manager)
        feat_out6 = self.relu_3d(out6)
        out_cat6 = ME.cat(feat_out6, out_b1p2)
        out6 = self.block7(out_cat6)

        # Linking @ p2
        V_B, C, H, W = p2.shape
        links_current_level = links.clone()
        links_current_level[:, 1:3, :] = ((H - 1.) / (h - 1.) * links_current_level[:, 1:3, :].float()).int()
        fused_3d_p2, fused_2d_p2 = self.linker_p2(p2, out6, links_current_level, init_3d_data=sparse_3d)

        # feat_3d = self.layer8_3d(ME.cat(fused_3d_p2, out_b1p2))
        out7 = self.convtr7p2s2(fused_3d_p2)
        out7 = self.bntr7(out7)
        feat_out7 = self.relu_3d(out7)
        out_cat7 = ME.cat(feat_out7, out_p1)
        out7 = self.block8(out_cat7)

        # Res
        # pdb.set_trace()
        res_2d = self.cls_2d(fused_2d_p2)
        res_2d = F.interpolate(res_2d, size=(h, w), mode='bilinear', align_corners=True)
        outputs_seg8_2d = torch.einsum('bqc,bchw->bqhw', mask_embed_2d, res_2d)
        outputs_seg_list_2d.append(outputs_seg8_2d)

        res_3d = self.cls_3d(out7)
        outputs_seg8 = self.mul_mask_voxel(mask_embed, res_3d)
        outputs_seg_list.append(outputs_seg8)

        predictions['pred_masks'] = outputs_seg_list[-1]
        V_B, C, H, W = outputs_seg8_2d.shape
        predictions_2d['pred_masks'] = outputs_seg_list_2d[-1].view(self.viewNum, b, C, H, W).permute(1, 2, 3, 4, 0)
        predictions_2d['aux_pred'] = outputs_seg_list_2d[:-1]

        if targets_mid_3d is not None:
            predictions["aux_pred"] = []
            predictions["aux_gt"] = []
            outputs_coords_list = [out_b4p16.C, out_b3p8.C, out_b2p4.C, out_b1p2.C]
            for i in range(len(targets_mid_3d)):
                outputs_segfeat_rearrange, targetsfeat_rearrange = self.get_rearrange_targets(outputs_seg_list[i], outputs_coords_list[i], targets_mid_3d[i])
                predictions["aux_pred"].append(outputs_segfeat_rearrange)
                predictions["aux_gt"].append(targetsfeat_rearrange)

        return predictions, predictions_2d

    def get_rearrange_targets(self, seg_F, seg_C, target_class):
        assert len(seg_F) == len(target_class)
        base = seg_C.max() + 1
        base2dec_gt = target_class.C[:, 0] * (base ** 3) + target_class.C[:, 1] * (base ** 2) + target_class.C[:, 2] * base + target_class.C[:, 3]
        base2dec_pred = seg_C[:, 0] * (base ** 3) + seg_C[:, 1] * (base ** 2) + seg_C[:, 2] * base + seg_C[:, 3]

        _, idx_gt = torch.sort(base2dec_gt)
        _, idx_pred = torch.sort(base2dec_pred)

        seg_rearrange_F = seg_F[idx_pred]
        seg_rearrange_C = seg_C[idx_pred]
        target_class_F = target_class.F[idx_gt]
        target_class_C = target_class.C[idx_gt]
        assert torch.equal(target_class_C, seg_rearrange_C)

        return seg_rearrange_F, target_class_F
    