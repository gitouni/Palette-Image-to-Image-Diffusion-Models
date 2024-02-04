import torch
import numpy as np
from PIL import Image
import argparse
import os
import shutil

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dir",type=str,default="debug/fig")
    parser.add_argument("--tensor",type=str,default="debug/tensor")
    return parser.parse_args()


if __name__ == "__main__":
    args = options()
    if os.path.exists(args.debug_dir):
        shutil.rmtree(args.debug_dir)
    os.makedirs(args.debug_dir)
    if os.path.isfile(args.tensor):
        data = torch.load(args.tensor, map_location='cpu')
        n_img = data.shape[0]
        for batch_i in range(n_img):
            img_np = np.transpose(data[batch_i,...].numpy(), (1, 2, 0))  # C, H ,W -> H, W, C
            img_uint8 = np.array(((img_np+1) * 127.5).round(), dtype=np.uint8)
            Image.fromarray(img_uint8).save(os.path.join(args.debug_dir, "{:04d}.png".format(batch_i)))
    elif os.path.isdir(args.tensor):
        tensor_files = os.listdir(args.tensor)
        for file in tensor_files:
            full_path = os.path.join(args.tensor, file)
            data = torch.load(full_path, map_location='cpu')
            img_np = np.transpose(data.numpy(), (1, 2, 0))  # C, H ,W -> H, W, C
            img_uint8 = np.array(((img_np+1) * 127.5).round(), dtype=np.uint8)
            Image.fromarray(img_uint8).save(os.path.join(args.debug_dir, "{}.png".format(os.path.splitext(file)[0])))
    else:
        raise RuntimeError("tensor para must be a file or dir, find {}".format(args.tensor))