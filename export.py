import os
import cv2
import re
import argparse
from tqdm import tqdm
import shutil
def refresh_dir(dirname:str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str,default="experiments/test_inpainting_tactile_240109_040516/results/test/0/")
    parser.add_argument("--input_fmt",type=str,default="Out_*")
    parser.add_argument("--resize",type=bool,default=False)
    parser.add_argument("--target_resol",type=int,nargs=2,default=[256,256],help="width,height")
    parser.add_argument("--output_dir",type=str,default="dataset/slip/nail1/Palette")
    parser.add_argument("--save_fmt",type=str,default=".png")
    args = parser.parse_args()
    refresh_dir(args.output_dir)
    input_file_list = os.listdir(args.input_dir)
    input_file_list = [file for file in input_file_list if re.match(args.input_fmt, file) is not None]
    input_file_list.sort()
    for input_file in tqdm(input_file_list):
        img = cv2.imread(os.path.join(args.input_dir, input_file))
        if args.resize:
            img = cv2.resize(img, args.target_resol, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.output_dir, os.path.splitext(input_file)[0]+args.save_fmt), img)
        
