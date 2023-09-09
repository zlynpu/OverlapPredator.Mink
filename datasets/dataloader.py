import MinkowskiEngine as ME
import torch
from lib.utils import load_obj
from datasets.indoor import IndoorDataset
from datasets.kitti import KITTIDataset
import numpy as np


def collate_pair_fn(list_data):
    src_xyz, tgt_xyz, src_coords, tgt_coords, src_feats, tgt_feats, matching_inds, rot, trans, scale, p_image, q_image = list(zip(*list_data))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')
    
    # prepare inputs for FCGF
    src_batch_C, src_batch_F = ME.utils.sparse_collate(src_coords, src_feats)
    tgt_batch_C, tgt_batch_F = ME.utils.sparse_collate(tgt_coords, tgt_feats)

    # concatenate xyz
    src_xyz = torch.cat(src_xyz, 0).float()
    tgt_xyz = torch.cat(tgt_xyz, 0).float()

    # add batch indice to matching_inds and image
    matching_inds_batch = []
    p_image_batch, q_image_batch = [], []
    len_batch = []
    curr_start_ind = torch.zeros((1,2))
    for batch_id, _ in enumerate(matching_inds):
        N0 = src_coords[batch_id].shape[0]
        N1 = tgt_coords[batch_id].shape[0]
        matching_inds_batch.append(matching_inds[batch_id]+curr_start_ind)
        len_batch.append([N0,N1])

        curr_start_ind[0,0]+=N0
        curr_start_ind[0,1]+=N1
        
        # image batch
        p_image_batch.append(to_tensor(p_image[batch_id][None,:,:,:]))
        q_image_batch.append(to_tensor(q_image[batch_id][None,:,:,:]))   

    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
    p_image_batch = torch.cat(p_image_batch,0).float()
    q_image_batch = torch.cat(q_image_batch,0).float()

    return {
        'pcd_src': src_xyz,
        'pcd_tgt': tgt_xyz,
        'src_C': src_batch_C,
        'src_F': src_batch_F,
        'tgt_C': tgt_batch_C,
        'tgt_F': tgt_batch_F,
        'correspondences': matching_inds_batch,
        'len_batch': len_batch,
        'rot': rot[0],
        'trans': trans[0],
        'scale': scale[0],
        'image0': p_image_batch,
        'image1': q_image_batch,
    }



def get_datasets(config):
    if(config.dataset=='indoor'):
        info_train = load_obj(config.train_info)
        info_val = load_obj(config.val_info)
        info_benchmark = load_obj(f'configs/indoor/{config.benchmark}.pkl')

        train_set = IndoorDataset(info_train,config,data_augmentation=True)
        val_set = IndoorDataset(info_val,config,data_augmentation=False)
        benchmark_set = IndoorDataset(info_benchmark,config, data_augmentation=False)
    elif(config.dataset == 'kitti'):
        train_set = KITTIDataset(config,'train',data_augmentation=True)
        val_set = KITTIDataset(config,'val',data_augmentation=False)
        benchmark_set = KITTIDataset(config, 'test',data_augmentation=False)
    else:
        raise NotImplementedError

    return train_set, val_set, benchmark_set
