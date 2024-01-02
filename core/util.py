import random
import numpy as np
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
from typing import Iterable, List, Tuple

def patchidx(target_size:Iterable[int], patch_size:Iterable[int], overlap:Iterable[int]):
    Hindex = list(range(0, target_size[0] - overlap[0], patch_size[0] - overlap[0]))
    Windex = list(range(0, target_size[1] - overlap[1], patch_size[1] - overlap[1]))
    Hindex[-1] = target_size[0] - patch_size[0]
    Windex[-1] = target_size[1] - patch_size[1]
    return Hindex, Windex

def img2patch(img:torch.Tensor, patch_size:Iterable[int], overlap:Iterable[int]) -> List[torch.Tensor]:
	if len(img.shape) == 3:
		H, W = img.shape[1], img.shape[0]
	elif len(img.shape) == 2:
		H, W = img.shape[0], img.shape[1]
	Hindex, Windex = patchidx((H,W), patch_size, overlap)
	patches = []
	for hi in Hindex:
		for wi in Windex:
			patches.append(img[...,hi:hi+256, wi:wi+256])
	return patches

def patch2img(patches:List[torch.Tensor], Hindex:Iterable[int], Windex:Iterable[int], target_size:Tuple[int]) -> torch.Tensor:
    img = torch.zeros(target_size).to(patches[0])
    count = torch.zeros(img.shape[-2:]).to(img)
    for patch_idx, (hi,wi) in enumerate(zip(Hindex, Windex)):
        img[...,hi:hi+256, wi:wi+256] += patches[patch_idx]
        count[hi:hi+256, wi:wi+256] += 1
    return (img / count).to(img)
            

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()

def postprocess(images):
	return [tensor2img(image) for image in images]


def set_seed(seed, gl_seed=0):
	"""  set random seed, gl_seed used in worker_init_fn function """
	if seed >=0 and gl_seed>=0:
		seed += gl_seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
	if seed >=0 and gl_seed>=0:  # slower, more reproducible
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:  # faster, less reproducible
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True

def set_gpu(args, distributed=False, rank=0):
	""" set parameter to gpu or ddp """
	if args is None:
		return None
	if distributed and isinstance(args, torch.nn.Module):
		return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
	else:
		return args.cuda()
		
def set_device(args, distributed=False, rank=0):
	""" set parameter to gpu or cpu """
	if torch.cuda.is_available():
		if isinstance(args, list):
			return (set_gpu(item, distributed, rank) for item in args)
		elif isinstance(args, dict):
			return {key:set_gpu(args[key], distributed, rank) for key in args}
		else:
			args = set_gpu(args, distributed, rank)
	return args

