import argparse
import os
from tqdm import tqdm
from utils.utils import refresh_dir
import numpy as np
import shutil

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--input_root",type=str,default="dataset/slip")
    io_parser.add_argument("--copied_subdirs",type=str,nargs="+", default=["markered", "marker"])
    io_parser.add_argument("--sampled_num",type=int,default=3)
    io_parser.add_argument("--output_dir",type=str,default="dataset/slip_sp")
    io_parser.add_argument("--save_fmt",type=str,default=".png")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    refresh_dir(args.output_dir)
    input_subdirs = sorted(os.listdir(args.input_root))
    for subsubdir in args.copied_subdirs:
        os.makedirs(os.path.join(args.output_dir, subsubdir))
    for subsubdir in args.copied_subdirs:
        file_index = 0
        for subdir in tqdm(input_subdirs, desc='data selection'):
            input_dir = os.path.join(args.input_root, subdir, subsubdir)
            output_dir = os.path.join(args.output_dir, subsubdir)
            img_list = list(sorted(os.listdir(input_dir)))
            num_file = len(img_list)
            num_index = np.linspace(0, num_file, args.sampled_num, endpoint=False).astype(np.int32)
            for file_idx in num_index:
                shutil.copyfile(os.path.join(input_dir, img_list[file_idx]), os.path.join(output_dir, "{:06d}{}".format(file_index, args.save_fmt)))
                file_index += 1