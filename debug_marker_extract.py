import argparse
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from utils.utils import find_marker_debug
import shutil
import os

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--debug_dir",type=str,default="debug_mask")
    io_parser.add_argument("--tgt_file",type=str,default="dataset/mkrand_256/img/0000.bmp")
    marker_parser = parser.add_argument_group()
    marker_parser.add_argument("--diff_min",type=int,default=144)
    marker_parser.add_argument("--diff_max",type=int,default=255)
    marker_parser.add_argument("--glength",type=float,default=30)
    marker_parser.add_argument("--gsig",type=float,default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    img = np.array(Image.open(args.tgt_file).convert("RGB"))
    gray, im_blur_3, im_blur_8, im_blur_sub, blur_mask, template, nrmcrimg, mask = \
        find_marker_debug(img, mask_range=(args.diff_min, args.diff_max), gkern_length=args.glength, gkern_sig=args.gsig)
    if os.path.exists(args.debug_dir):
        shutil.rmtree(args.debug_dir)
    os.makedirs(args.debug_dir)
    Image.fromarray(gray).save(os.path.join(args.debug_dir, "gray.png"))
    Image.fromarray(im_blur_3).save(os.path.join(args.debug_dir, "blur_3.png"))
    Image.fromarray(im_blur_8).save(os.path.join(args.debug_dir, "blur_8.png"))
    Image.fromarray(im_blur_sub).save(os.path.join(args.debug_dir, "blur_sub.png"))
    Image.fromarray(blur_mask).save(os.path.join(args.debug_dir, "blur_mask.png"))
    plt.figure(figsize=(4.8,4.8))
    plt.imshow(template)
    plt.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0, pad=0)
    plt.savefig(os.path.join(args.debug_dir, "template.png"))
    plt.clf()
    plt.figure(figsize=(4.8,4.8))
    plt.imshow(nrmcrimg)
    plt.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0, pad=0)
    plt.savefig(os.path.join(args.debug_dir, "nrmcrimg.png"))
    Image.fromarray(mask*255).save(os.path.join(args.debug_dir, "mask.png"))