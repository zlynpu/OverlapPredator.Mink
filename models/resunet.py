import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from models.common import get_norm
from models.gcn import GCN
# from models.attention_fusion import AttentionFusion
from models.Local_Fusion import Local_Atten_Fusion_Conv
from models.Img_Encoder import ImageEncoder
from models.Img_Decoder import ImageDecoder
import torch.nn.functional as F
from torch.nn.functional import grid_sample

from models.residual_block import get_block
import torch.nn as nn


class ResUNet2(ME.MinkowskiNetwork):
	NORM_TYPE = None
	BLOCK_NORM_TYPE = 'BN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 32, 64, 64, 128]

	FUSION_IMG_CHANNELS = [64, 64, 32]
	FUSION_POINT_CHANNELS = [64, 128, 64]

	# To use the model, must call initialize_coords before forward pass.
	# Once data is processed, call clear to reset the model before calling initialize_coords
	def __init__(self,config,D=3):
		ME.MinkowskiNetwork.__init__(self, D)
		NORM_TYPE = self.NORM_TYPE
		BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
		CHANNELS = self.CHANNELS
		TR_CHANNELS = self.TR_CHANNELS
		FUSION_IMG_CHANNELS = self.FUSION_IMG_CHANNELS
		FUSION_POINT_CHANNELS = self.FUSION_POINT_CHANNELS
		bn_momentum = config.bn_momentum
		self.normalize_feature = config.normalize_feature
		self.voxel_size = config.voxel_size

		self.conv1 = ME.MinkowskiConvolution(
			in_channels=config.in_feats_dim,
			out_channels=CHANNELS[1],
			kernel_size=config.conv1_kernel_size,
			stride=1,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

		self.block1 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

		self.conv2 = ME.MinkowskiConvolution(
			in_channels=CHANNELS[1],
			out_channels=CHANNELS[2],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.block2 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.conv3 = ME.MinkowskiConvolution(
			in_channels=CHANNELS[2],
			out_channels=CHANNELS[3],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.block3 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.conv4 = ME.MinkowskiConvolution(
			in_channels=CHANNELS[3],
			out_channels=CHANNELS[4],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

		self.block4 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

		###############
		# image-pointcloud fusion model
		self.fusion_attention = nn.ModuleList()
		for i in range(len(FUSION_IMG_CHANNELS)): 
			self.fusion_attention.append(Local_Atten_Fusion_Conv(inplanes_I=FUSION_IMG_CHANNELS[i], 
                                                           		inplanes_P=FUSION_POINT_CHANNELS[i],
                                                           		outplanes=FUSION_POINT_CHANNELS[i]))

		# adapt input tensor here
		self.conv4_tr = ME.MinkowskiConvolutionTranspose(
			in_channels = config.gnn_feats_dim + 2,
			# in_channels = config.gnn_feats_dim,
			out_channels=TR_CHANNELS[4],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

		self.block4_tr = get_block(
			BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

		self.conv3_tr = ME.MinkowskiConvolutionTranspose(
			in_channels=CHANNELS[3] + TR_CHANNELS[4],
			out_channels=TR_CHANNELS[3],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.block3_tr = get_block(
			BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.conv2_tr = ME.MinkowskiConvolutionTranspose(
			in_channels=CHANNELS[2] + TR_CHANNELS[3],
			out_channels=TR_CHANNELS[2],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.block2_tr = get_block(
			BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.conv1_tr = ME.MinkowskiConvolution(
			in_channels=CHANNELS[1] + TR_CHANNELS[2],
			out_channels=TR_CHANNELS[1],
			kernel_size=1,
			stride=1,
			dilation=1,
			bias=False,
			dimension=D)

		self.final = ME.MinkowskiConvolution(
			in_channels=TR_CHANNELS[1],
			out_channels=config.out_feats_dim + 2,
			# out_channels=config.out_feats_dim,
			kernel_size=1,
			stride=1,
			dilation=1,
			bias=True,
			dimension=D)


		#############
		# Overlap attention module
		self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
		self.bottle = nn.Conv1d(CHANNELS[4], config.gnn_feats_dim,kernel_size=1,bias=True)
		self.gnn = GCN(config.num_head,config.gnn_feats_dim, config.dgcnn_k, config.nets)
		self.proj_gnn = nn.Conv1d(config.gnn_feats_dim,config.gnn_feats_dim,kernel_size=1, bias=True)
		self.proj_score = nn.Conv1d(config.gnn_feats_dim,1,kernel_size=1,bias=True)


		################
		# preprocess image
		self.img_encoder = ImageEncoder()
		self.img_decoder = ImageDecoder()
		
	def local_fusion(self, coord, feature, image_feature, image_shape, extrinsic, intrinsic, id, voxel_size):
    	# batch ----- batch
		lengths = []
		max_batch = torch.max(coord[:, 0])
		for i in range(max_batch + 1):
			length = torch.sum(coord[:, 0] == i)
			lengths.append(length)
		
		device = torch.device('cuda')
		fusion_feature_batch = []
		start = 0
		end = 0
		batch_id = 0
    
		for length in lengths:
			end += length
			point_coord = coord[start:end, 1:] * voxel_size
			point_feature = feature[start:end, :].unsqueeze(0)
			point_feature = point_feature.permute(0,2,1)
			#############################
			# 1. project point to image feature map
			point_z = point_coord[:, -1]
			one = torch.ones((point_coord.shape[0],1)).to(device)
			point_in_lidar = torch.cat([point_coord, one],1).t()

			point_in_camera = extrinsic[batch_id, :, :].mm(point_in_lidar)
			point_in_image = intrinsic[batch_id, :, :].mm(point_in_camera)/point_z
			point_in_image = point_in_image.t()
			point_in_image[:, -1] = point_z
			point_in_image[:, 0] = point_in_image[:, 0] * 2 / image_shape[batch_id, 0] - 1
			point_in_image[:, 1] = point_in_image[:, 1] * 2 / image_shape[batch_id, 1] - 1
			point_in_image = point_in_image.unsqueeze(0)

			feature_map = image_feature[batch_id, :, :, :].unsqueeze(0)
			xy = point_in_image[:, :, :-1].unsqueeze(1)
			# 2. grid sample image feature map according to projection
			img_feature = grid_sample(feature_map, xy, align_corners=False)
			img_feature = img_feature.squeeze(2)

			# 3. fuse the point feature and the image map using attention model
			fusion_feature = self.fusion_attention[id](point_features=point_feature, img_features=img_feature)

			fusion_feature_batch.append(fusion_feature)
			start += length
			batch_id += 1
    
		fusion_feature_batch = torch.cat(fusion_feature_batch,2)
			
		return fusion_feature_batch

	def forward(self, stensor_src, stensor_tgt, src_image, tgt_image, image_shape, extrinsic, intrinsic):
		################################
		# encode image(only consider single scale)
		src_image1, src_image2, src_image3 = self.img_encoder(src_image)
		tgt_image1, tgt_image2, tgt_image3 = self.img_encoder(tgt_image)

		################################
		# decode multi-scale image feature and fusion
		src_image_fusion = self.img_decoder(src_image1, src_image2, src_image3)
		tgt_image_fusion = self.img_decoder(tgt_image1, tgt_image2, tgt_image3)

		################################
		# encode src
		src_s1 = self.conv1(stensor_src)
		src_s1 = self.norm1(src_s1)
		src_s1 = self.block1(src_s1)
		src = MEF.relu(src_s1)

		src_s2 = self.conv2(src)
		src_s2 = self.norm2(src_s2)
		src_s2 = self.block2(src_s2)
		src = MEF.relu(src_s2)
		src_fusion2 = self.local_fusion(coord=src.C, feature=src.F, image_feature=src_image1, 
								  		image_shape=image_shape, extrinsic=extrinsic, intrinsic=intrinsic, 
										id=0, voxel_size=self.voxel_size)
    	# (B, 64, N)
		src_fusion2 = src_fusion2.reshape(-1,src_fusion2.shape[-1])
    	# (64, N*B)
		src._F = src_fusion2.permute(1,0)
		del src_fusion2

		src_s4 = self.conv3(src)
		src_s4 = self.norm3(src_s4)
		src_s4 = self.block3(src_s4)
		src = MEF.relu(src_s4)
		src_fusion4 = self.local_fusion(coord=src.C, feature=src.F, image_feature=src_image1, 
								  		image_shape=image_shape, extrinsic=extrinsic, intrinsic=intrinsic, 
										id=1, voxel_size=self.voxel_size)
    	# (B, 64, N)
		src_fusion4 = src_fusion4.reshape(-1,src_fusion4.shape[-1])
    	# (64, N*B)
		src._F = src_fusion4.permute(1,0)
		del src_fusion4

		src_s8 = self.conv4(src)
		src_s8 = self.norm4(src_s8)
		src_s8 = self.block4(src_s8)
		src = MEF.relu(src_s8)

		################################
		# encode tgt
		tgt_s1 = self.conv1(stensor_tgt)
		tgt_s1 = self.norm1(tgt_s1)
		tgt_s1 = self.block1(tgt_s1)
		tgt = MEF.relu(tgt_s1)

		tgt_s2 = self.conv2(tgt)
		tgt_s2 = self.norm2(tgt_s2)
		tgt_s2 = self.block2(tgt_s2)
		tgt = MEF.relu(tgt_s2)
		tgt_fusion2 = self.local_fusion(coord=tgt.C, feature=tgt.F, image_feature=tgt_image1, 
								  		image_shape=image_shape, extrinsic=extrinsic, intrinsic=intrinsic, 
										id=0, voxel_size=self.voxel_size)
    	# (B, 64, N)
		tgt_fusion2 = tgt_fusion2.reshape(-1,tgt_fusion2.shape[-1])
    	# (64, N*B)
		tgt._F = tgt_fusion2.permute(1,0)
		del tgt_fusion2

		tgt_s4 = self.conv3(tgt)
		tgt_s4 = self.norm3(tgt_s4)
		tgt_s4 = self.block3(tgt_s4)
		tgt = MEF.relu(tgt_s4)
		tgt_fusion4 = self.local_fusion(coord=tgt.C, feature=tgt.F, image_feature=tgt_image1, 
								  		image_shape=image_shape, extrinsic=extrinsic, intrinsic=intrinsic, 
										id=1, voxel_size=self.voxel_size)
    	# (B, 64, N)
		tgt_fusion4 = tgt_fusion4.reshape(-1,tgt_fusion4.shape[-1])
    	# (64, N*B)
		tgt._F = tgt_fusion4.permute(1,0)
		del tgt_fusion4

		tgt_s8 = self.conv4(tgt)
		tgt_s8 = self.norm4(tgt_s8)
		tgt_s8 = self.block4(tgt_s8)
		tgt = MEF.relu(tgt_s8)

		# ################################
		# overlap attention module
		# empirically, when batch_size = 1, out.C[:,1:] == out.coordinates_at(0)		
		src_feats = src.F.transpose(0,1)[None,:]  #[1, C, N]
		tgt_feats = tgt.F.transpose(0,1)[None,:]  #[1, C, N]
		src_pcd, tgt_pcd = src.C[:,1:] * self.voxel_size, tgt.C[:,1:] * self.voxel_size

		# 1. project the bottleneck feature
		src_feats, tgt_feats = self.bottle(src_feats), self.bottle(tgt_feats)

		# 2. apply GNN to communicate the features and get overlap scores
		src_feats, tgt_feats= self.gnn(src_pcd.transpose(0,1)[None,:], tgt_pcd.transpose(0,1)[None,:],src_feats, tgt_feats)

		src_feats, src_scores = self.proj_gnn(src_feats), self.proj_score(src_feats)[0].transpose(0,1)
		tgt_feats, tgt_scores = self.proj_gnn(tgt_feats), self.proj_score(tgt_feats)[0].transpose(0,1)
		

		# 3. get cross-overlap scores
		src_feats_norm = F.normalize(src_feats, p=2, dim=1)[0].transpose(0,1)
		tgt_feats_norm = F.normalize(tgt_feats, p=2, dim=1)[0].transpose(0,1)
		inner_products = torch.matmul(src_feats_norm, tgt_feats_norm.transpose(0,1))
		temperature = torch.exp(self.epsilon) + 0.03
		src_scores_x = torch.matmul(F.softmax(inner_products / temperature ,dim=1) ,tgt_scores)
		tgt_scores_x = torch.matmul(F.softmax(inner_products.transpose(0,1) / temperature,dim=1),src_scores)

		# 4. update sparse tensor
		src_feats = torch.cat([src_feats[0].transpose(0,1), src_scores, src_scores_x], dim=1)
		tgt_feats = torch.cat([tgt_feats[0].transpose(0,1), tgt_scores, tgt_scores_x], dim=1)
		src = ME.SparseTensor(src_feats, 
			coordinate_map_key=src.coordinate_map_key,
			coordinate_manager=src.coordinate_manager)

		tgt = ME.SparseTensor(tgt_feats,
			coordinate_map_key=tgt.coordinate_map_key,
			coordinate_manager=tgt.coordinate_manager)


		################################
		# decoder src
		src = self.conv4_tr(src)
		src = self.norm4_tr(src)
		src = self.block4_tr(src)
		src_s4_tr = MEF.relu(src)

		src = ME.cat(src_s4_tr, src_s4)
		del src_s4_tr
		del src_s4

		src = self.conv3_tr(src)
		src = self.norm3_tr(src)
		src = self.block3_tr(src)
		src_s2_tr = MEF.relu(src)

		src = ME.cat(src_s2_tr, src_s2)
		del src_s2_tr
		del src_s2

		src = self.conv2_tr(src)
		src = self.norm2_tr(src)
		src = self.block2_tr(src)
		src_s1_tr = MEF.relu(src)

		src = ME.cat(src_s1_tr, src_s1)
		del src_s1
		del src_s1_tr

		src = self.conv1_tr(src)
		src = MEF.relu(src)
		src_fusion_final = self.local_fusion(coord=src.C, feature=src.F, image_feature=src_image_fusion, 
									   		image_shape=image_shape, extrinsic=extrinsic, intrinsic=intrinsic, 
											id=2, voxel_size=self.voxel_size)
		src_fusion_final = src_fusion_final.reshape(-1,src_fusion_final.shape[-1])
		src._F = src_fusion_final.permute(1,0)
		src = self.final(src)
		del src_fusion_final

		################################
		# decoder tgt
		tgt = self.conv4_tr(tgt)
		tgt = self.norm4_tr(tgt)
		tgt = self.block4_tr(tgt)
		tgt_s4_tr = MEF.relu(tgt)

		tgt = ME.cat(tgt_s4_tr, tgt_s4)
		del tgt_s4_tr
		del tgt_s4

		tgt = self.conv3_tr(tgt)
		tgt = self.norm3_tr(tgt)
		tgt = self.block3_tr(tgt)
		tgt_s2_tr = MEF.relu(tgt)

		tgt = ME.cat(tgt_s2_tr, tgt_s2)
		del tgt_s2_tr
		del tgt_s2

		tgt = self.conv2_tr(tgt)
		tgt = self.norm2_tr(tgt)
		tgt = self.block2_tr(tgt)
		tgt_s1_tr = MEF.relu(tgt)

		tgt = ME.cat(tgt_s1_tr, tgt_s1)
		del tgt_s1_tr
		del tgt_s1

		tgt = self.conv1_tr(tgt)
		tgt = MEF.relu(tgt)
		tgt_fusion_final = self.local_fusion(coord=tgt.C, feature=tgt.F, image_feature=tgt_image_fusion, 
									   		image_shape=image_shape, extrinsic=extrinsic, intrinsic=intrinsic, 
											id=2, voxel_size=self.voxel_size)
		tgt_fusion_final = tgt_fusion_final.reshape(-1,tgt_fusion_final.shape[-1])
		tgt._F = tgt_fusion_final.permute(1,0)
		tgt = self.final(tgt)
		del tgt_fusion_final

		################################
		# output features and scores
		sigmoid = nn.Sigmoid()
		src_feats, src_overlap, src_saliency = src.F[:,:-2], src.F[:,-2], src.F[:,-1]
		tgt_feats, tgt_overlap, tgt_saliency = tgt.F[:,:-2], tgt.F[:,-2], tgt.F[:,-1]

		src_overlap= torch.clamp(sigmoid(src_overlap.view(-1)),min=0,max=1)
		src_saliency = torch.clamp(sigmoid(src_saliency.view(-1)),min=0,max=1)
		tgt_overlap = torch.clamp(sigmoid(tgt_overlap.view(-1)),min=0,max=1)
		tgt_saliency = torch.clamp(sigmoid(tgt_saliency.view(-1)),min=0,max=1)
		
		src_feats = F.normalize(src_feats, p=2, dim=1)
		tgt_feats = F.normalize(tgt_feats, p=2, dim=1)

		scores_overlap = torch.cat([src_overlap, tgt_overlap], dim=0)
		scores_saliency = torch.cat([src_saliency, tgt_saliency], dim=0)

		return src_feats,  tgt_feats, scores_overlap, scores_saliency

		####################
		# IMFNet
		# return ME.SparseTensor(
        #   src.F / torch.norm(src.F, p=2, dim=1, keepdim=True),
        #   coordinate_map_key=src.coordinate_map_key,
        #   coordinate_manager=src.coordinate_manager
      	# ), ME.SparseTensor(
        #   tgt.F / torch.norm(tgt.F, p=2, dim=1, keepdim=True),
        #   coordinate_map_key=tgt.coordinate_map_key,
        #   coordinate_manager=tgt.coordinate_manager
      	# )



class ResUNetBN2(ResUNet2):
  	NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
	NORM_TYPE = 'BN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
	NORM_TYPE = 'IN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 64, 64, 64, 128]
	BLOCK_NORM_TYPE = 'IN'

	# CHANNELS = [None, 64, 128, 256, 512]
	# TR_CHANNELS = [None, 64, 128, 128, 256]


class ResUNetBN2D(ResUNet2):
	NORM_TYPE = 'BN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
	NORM_TYPE = 'BN'
	CHANNELS = [None, 128, 128, 128, 256]
	TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'
