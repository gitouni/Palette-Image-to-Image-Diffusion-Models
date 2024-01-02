import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import cv2
import os
import torch
import numpy as np
from pathlib import Path
from queue import Queue
from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)
import sys
sys.path.append(os.path.join("..", os.path.dirname(__file__)))
sys.path.append(os.path.join("../..", os.path.dirname(__file__)))
from core.util import patchidx


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path, mode='RGB'):
    return Image.open(path).convert(mode)

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, glob_fmt="", mask_root="", mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        if not glob_fmt:
            imgs = make_dataset(data_root)
        else:
            imgs = [str(path) for path in Path(data_root).glob(glob_fmt)]
            imgs.sort()
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),  # height ,width
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        if "file" in self.mask_mode:
            assert(os.path.isdir(mask_root))
            self.mask_root = mask_root
            self.mask_files = list(sorted(os.listdir(mask_root)))
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path:str = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask(index)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index:int, resize=True):
        mask_transfer = lambda mask: torch.from_numpy(mask).permute(2,0,1).float()
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif 'file' in self.mask_mode:
            mask_pil = cv2.imread(os.path.join(self.mask_root, self.mask_files[index]), cv2.IMREAD_GRAYSCALE)
            if resize:
                mask_pil = cv2.resize(mask_pil, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
                mask = np.zeros(self.image_size, dtype=np.uint8)
            else:
                mask = np.zeros(mask_pil.shape, dtype=np.uint8)
            mask[mask_pil > 0] = 1
            dilate_size = self.mask_config['mask_dilate']['size']
            dilate_iter = self.mask_config['mask_dilate']['iteration']
            
            if self.mask_mode == 'file':
                pass
            elif self.mask_mode == 'file_dilate':
                dilate_size = self.mask_config['mask_dilate']['size']
                dilate_iter = self.mask_config['mask_dilate']['iteration']
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_size), iterations=dilate_iter)
            elif self.mask_mode == 'file_dilate_delta':
                dilate_size = self.mask_config['mask_dilate']['size']
                dilate_iter = self.mask_config['mask_dilate']['iteration']
                dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_size), iterations=dilate_iter)
                mask = np.logical_and(dilated_mask, np.logical_not(mask))  # set subtraction
            elif self.mask_mode == "file_offset":
                offset = self.mask_config['mask_offset']
                mask_yx = np.nonzero(mask)
                new_mask_x = mask_yx[1] + offset[0]
                new_mask_y = mask_yx[0] + offset[1]
                h, w = mask.shape[:2]
                rev = (new_mask_x >=0) * (new_mask_x < w) * (new_mask_y >=0) * (new_mask_y < h)
                new_mask_x = new_mask_x[rev]
                new_mask_y = new_mask_y[rev]
                new_mask = np.zeros_like(mask)  # uint8
                new_mask[new_mask_y, new_mask_x] = 1
                mask = np.logical_and(new_mask, np.logical_not(mask))  # set subtraction
            else:
                raise NotImplementedError(
                    f'Mask mode {self.mask_mode} has not been implemented.')
            mask = mask[...,None]  # (H, W ,1)

        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')

        return mask_transfer(mask)

class CropInpaintDataset(InpaintDataset):
    def __init__(self, data_root, glob_fmt="", mask_root="", mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        super().__init__(data_root, glob_fmt, mask_root, mask_config, data_len, image_size, loader)
        self.tfs = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.crop_tfs = transforms.RandomCrop((image_size[0], image_size[1]))  # height ,width

    def __getitem__(self, index):
        ret = {}
        path:str = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask(index)
        concat = torch.cat((img, mask), dim=0)  # (3,H,W) + (1,H,W)
        crop_concat = self.crop_tfs(concat)
        img, mask = crop_concat[:3,...], crop_concat[[-1],...]
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret
        
    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index:int):
        return super().get_mask(index, resize=False)
    
class PatchInapintDataset(InpaintDataset):
    def __init__(self, data_root, glob_fmt="", mask_root="", mask_config={}, data_len=-1, buffer_size=5, image_size=[256, 256], overlap=[30,30], loader=pil_loader):
        super().__init__(data_root, glob_fmt, mask_root, mask_config, data_len, image_size, loader)
        self.tfs = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        ex_img = pil_loader(self.imgs[0])
        self.image_size = ex_img.height, ex_img.width
        self.patch_size = image_size
        self.patch_hidx, self.patch_widx = patchidx(self.image_size, self.patch_size, overlap)
        self.patch_idx = []
        for hidx in self.patch_hidx:
            for widx in self.patch_widx:
                self.patch_idx.append((hidx, widx))
        self.patch_num = len(self.patch_idx)
        self.idx_buffer = Queue(buffer_size)  # adaptive to multiprocessing
        self.buffer = dict()
    
    def __getitem__(self, index):
        file_index = int(index / self.patch_num)
        ret = {}
        ret['file_index'] = file_index
        ret['patch_idx'] = index % self.patch_num
        if file_index not in self.buffer.keys():
            path:str = self.imgs[file_index]
            img = self.tfs(self.loader(path))
            mask = self.get_mask(file_index) 
            cond_image = img*(1. - mask) + mask*torch.randn_like(img)
            mask_img = img*(1. - mask) + mask
            path = path.rsplit("/")[-1].rsplit("\\")[-1]
            self.buffer[file_index] = dict(img=img, cond=cond_image, mask=mask, mask_img=mask_img, path=path)
            self.idx_buffer.put(file_index)
            if self.idx_buffer.full():
                removed_idx = self.idx_buffer.get(timeout=1.0)
                self.buffer.pop(removed_idx)
        else:
            cache = self.buffer[file_index]
            img, cond_image, mask, mask_img, path = cache['img'], cache['cond'], cache['mask'], cache['mask_img'], cache['path']
        hidx, widx = self.patch_idx[ret['patch_idx']]
        dh = self.patch_size[0]
        dw = self.patch_size[1]
        ret['gt_image'] = img[...,hidx:hidx+dh, widx:widx+dw]
        ret['cond_image'] = cond_image[...,hidx:hidx+dh, widx:widx+dw]
        ret['mask_image'] = mask_img[...,hidx:hidx+dh, widx:widx+dw]
        ret['mask'] = mask[...,hidx:hidx+dh, widx:widx+dw]
        ret['path'] = path
        return ret
    
    def __len__(self):
        return super().__len__() * self.patch_num
    
    def get_mask(self, index:int):
        return super().get_mask(index, resize=False)
    
    @staticmethod
    def collate_fn(batch):
        return {
            'gt_image': torch.stack([x['gt_image'] for x in batch]),
            'cond_image': torch.stack([x['cond_image'] for x in batch]),
            'mask_image': torch.stack([x['mask_image'] for x in batch]),
            'mask': torch.stack([x['mask'] for x in batch]),
            'path': [x['path'] for x in batch],
            'file_index': [x['file_index'] for x in batch],
            'patch_idx': [x['patch_idx'] for x in batch]
        }

class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)  # (H, W ,C) - > (C, H ,W)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


