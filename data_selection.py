import os
import numpy as np
import argparse
from PIL import Image
import shutil
def refresh_dir(dirname:str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)
    
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_img",type=str,default="dataset/vitac/img")
    parser.add_argument("--src_mask",type=str,default="dataset/vitac/marker")
    parser.add_argument("--src_label",type=str,default="dataset/vitac/label.txt")
    parser.add_argument("--max_sample",type=int,default=5)
    parser.add_argument("--tgt_img",type=str, default="dataset/vitac/simg/")
    parser.add_argument("--tgt_mask",type=str, default="dataset/vitac/smask/")
    parser.add_argument("--tgt_label",type=str, default="dataset/vitac/slabel.txt")
    parser.add_argument("--tgt_img_size",type=int,nargs=2, default=[256,256])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    src_img_files = sorted(os.listdir(args.src_img))
    src_mask_files = sorted(os.listdir(args.src_mask))
    src_label = np.loadtxt(args.src_label,dtype=np.int32).flatten()
    unqiue_label = np.unique(src_label)
    unqiue_mask = [src_label == label for label in unqiue_label]
    index_arr = np.arange(len(src_img_files))
    unqiue_index = [index_arr[mask] for mask in unqiue_mask]
    selected_index = np.concatenate([np.linspace(idx_arr[0], idx_arr[-1], num=min(args.max_sample, len(idx_arr)), endpoint=True) for idx_arr in unqiue_index])
    selected_index = selected_index.astype(np.int32)
    refresh_dir(args.tgt_img)
    refresh_dir(args.tgt_mask)
    for index in selected_index:
        img = Image.open(os.path.join(args.src_img, src_img_files[index])).resize(args.tgt_img_size, resample=Image.Resampling.BICUBIC)
        marker = Image.open(os.path.join(args.src_mask, src_mask_files[index])).resize(args.tgt_img_size, resample=Image.Resampling.NEAREST)
        img.save(os.path.join(args.tgt_img, src_img_files[index]))
        marker.save(os.path.join(args.tgt_mask, src_mask_files[index]))
    sampled_label = src_label[selected_index]
    np.savetxt(args.tgt_label, sampled_label, fmt="%d")