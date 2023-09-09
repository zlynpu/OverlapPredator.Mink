import torch
import numpy as np
import MinkowskiEngine as ME
import time

def _hash(arr, M):
  if isinstance(arr, np.ndarray):
    N, D = arr.shape
  else:
    N, D = len(arr[0]), len(arr)

  hash_vec = np.zeros(N, dtype=np.int64)
  for d in range(D):
    if isinstance(arr, np.ndarray):
      hash_vec += arr[:, d] * M**d
    else:
      hash_vec += arr[d] * M**d
  return hash_vec


def extract_features(model,
                     src_xyz,
                     tgt_xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True,
                     src_image=None,
                     tgt_image=None
                     ):
  '''
  xyz is a N x 3 matrix
  rgb is a N x 3 matrix and all color must range from [0, 1] or None
  normal is a N x 3 matrix and all normal range from [-1, 1] or None

  if both rgb and normal are None, we use Nx1 one vector as an input

  if device is None, it tries to use gpu by default

  if skip_check is True, skip rigorous checks to speed up

  model = model.to(device)
  xyz, feats = extract_features(model, xyz)
  '''
  if is_eval:
    model.eval()

  if not skip_check:
    assert src_xyz.shape[1] == 3
    assert tgt_xyz.shape[1] == 3

    N0 = src_xyz.shape[0]
    N1 = tgt_xyz.shape[0]
    if rgb is not None:
      assert N0 == len(rgb)
      assert rgb.shape[1] == 3
      if np.any(rgb > 1):
        raise ValueError('Invalid color. Color must range from [0, 1]')

    if normal is not None:
      assert N0 == len(normal)
      assert normal.shape[1] == 3
      if np.any(normal > 1):
        raise ValueError('Invalid normal. Normal must range from [-1, 1]')

  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  feats0 = []
  feats1 = []
  if rgb is not None:
    # [0, 1]
    feats0.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats0.append(normal / 2)

  if rgb is None and normal is None:
    feats0.append(np.ones((len(src_xyz), 1)))
    feats1.append(np.ones((len(tgt_xyz), 1)))

  feats0 = np.hstack(feats0)
  feats1 = np.hstack(feats1)

  # Voxelize xyz and feats
  coords0 = np.floor(src_xyz / voxel_size)
  coords1 = np.floor(tgt_xyz / voxel_size)
  coords0, inds0 = ME.utils.sparse_quantize(coords0, return_index=True)
  coords1, inds1 = ME.utils.sparse_quantize(coords1, return_index=True)

  # coords,inds = coords[:5000],inds[:5000]
  # Convert to batched coords compatible with ME
  coords0 = ME.utils.batched_coordinates([coords0])
  coords1 = ME.utils.batched_coordinates([coords1])
  return_coords0 = src_xyz[inds0]
  return_coords1 = tgt_xyz[inds1]

  feats0 = feats0[inds0]
  feats1 = feats1[inds1]

  feats0 = torch.tensor(feats0, dtype=torch.float32)
  feats1 = torch.tensor(feats1, dtype=torch.float32)
  coords0 = torch.tensor(coords0, dtype=torch.int32)
  coords1 = torch.tensor(coords1, dtype=torch.int32)


  stensor0 = ME.SparseTensor(feats0, coordinates=coords0, device=device)
  stensor1 = ME.SparseTensor(feats1, coordinates=coords1, device=device)

  src_image = torch.as_tensor(src_image,dtype=torch.float32,device=device)
  tgt_image = torch.as_tensor(tgt_image,dtype=torch.float32,device=device)

  # start = time.time()
  result0, result1 = model(stensor0,stensor1,src_image,tgt_image)
  F0 = result0.F
  F1 = result1.F
  # end = time.time()
  # t = end - start

  return return_coords0,F0,return_coords1,F1
