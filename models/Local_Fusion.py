import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample
import numpy as np
import sys
import os

ROOT_DIR = os.path.abspath('/data1/zhangliyuan/code/PIFNet')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

class IA_Layer(nn.Module):
    def __init__(self, channels):
        # print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp))) #BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out


class Local_Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Local_Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

# def local_fusion(coord, feature, image_feature, image_shape, extrinsic, intrinsic,voxel_size=0.025):
#     # batch ----- batch
#     lengths = []
#     max_batch = torch.max(coord[:, 0])
#     for i in range(max_batch + 1):
#         length = torch.sum(coord[:, 0] == i)
#         lengths.append(length)

#     device = torch.device('cuda')
#     fusion_feature_batch = []
#     start = 0
#     end = 0
#     batch_id = 0
    
#     for length in lengths:
#         end += length
#         point_coord = coord[start:end, 1:] * voxel_size
#         point_feature = feature[start:end, :].unsqueeze(0)
#         point_feature = point_feature.permute(0,2,1)

#         point_z = point_coord[:, -1]
#         # print('pointcoord',point_coord.shape)
#         one = torch.ones((point_coord.shape[0],1)).to(device)
#         # print('one',one.shape)
#         point_in_lidar = torch.cat([point_coord, one],1).t()
#         # print('point_in_lidar',point_in_lidar.shape)

#         point_in_camera = extrinsic[batch_id, :, :].mm(point_in_lidar)
#         point_in_image = intrinsic[batch_id, :, :].mm(point_in_camera)/point_z
#         point_in_image = point_in_image.t()
#         point_in_image[:, -1] = point_z

#         point_in_image[:, 0] = point_in_image[:, 0] * 2 / image_shape[batch_id, 0] - 1
#         point_in_image[:, 1] = point_in_image[:, 1] * 2 / image_shape[batch_id, 1] - 1

#         point_in_image = point_in_image.unsqueeze(0)

#         feature_map = image_feature[batch_id, :, :, :].unsqueeze(0)
#         # print('feature_map_shape:',feature_map.shape)
#         xy = point_in_image[:, :, :-1].unsqueeze(1)
#         # print('xy:',xy.shape)
#         img_feature = grid_sample(feature_map, xy)
#         img_feature = img_feature.squeeze(2)
        
        
#         fusion_feature = torch.cat([img_feature, point_feature], dim=1)
#         # print('fusion_feature:', fusion_feature.shape)

#         fusion_feature_batch.append(fusion_feature)
#         start += length
#         batch_id += 1
    
#     fusion_feature_batch = torch.cat(fusion_feature_batch,2)
#     # print('fusion_feature_batch:',fusion_feature_batch.shape)
#     return fusion_feature_batch





