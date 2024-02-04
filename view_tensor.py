import torch
import numpy as np
from PIL import Image
import argparse
import os
import shutil

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dir",type=str,default="Fig/debug/test")
    parser.add_argument("--output_dir",type=str,default="Fig/img/test")
    return parser.parse_args()


if __name__ == "__main__":
    args = options()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    tensor_files = sorted(os.listdir(args.tensor_dir))
    for tensor_file in tensor_files:
        data:torch.Tensor = torch.load(os.path.join(args.tensor_dir, tensor_file), map_location='cpu')[0].squeeze().detach().numpy()
        if np.ndim(data) == 3:
            img_np = np.transpose(data, (1,2,0))
        else:
            img_np = data
        invalid = img_np == 0
        img_uint8 = np.array(((img_np+1) * 127.5).round(), dtype=np.uint8)
        img_uint8[invalid] = 0
        Image.fromarray(img_uint8).save(os.path.join(args.output_dir, "{}.png".format(os.path.splitext(tensor_file)[0])))