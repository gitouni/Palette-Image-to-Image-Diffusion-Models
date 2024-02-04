import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import re

def rmse(gt:np.ndarray, pred:np.ndarray):
    return np.sqrt(np.sum((gt-pred)**2, axis=-1))

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--marker_path",type=str,default="dataset/vitac/te_smarker/003641.png")
    io_parser.add_argument("--img_path",type=str,nargs="+",default=["dataset/vitac/te_NS/003641.bmp",
                                                                 "dataset/vitac/te_TELEA/003641.bmp",
                                                                 "dataset/vitac/te_Palette/Out_003641.bmp",
                                                                 "experiments/test_inpainting_tactile_231220_055533/results/test/0/Out_003641.bmp"])
    io_parser.add_argument("--method",type=str,nargs="+",default=["NS","TELEA","Palette","Palette_Semi"])
    io_parser.add_argument("--save_pattern",type=str,default=r"\d{6}")
    io_parser.add_argument("--save_dir",type=str,default="debug_cmp_nogt")
    vis_parser = parser.add_argument_group()
    vis_parser.add_argument("--canny_threshold",type=int,nargs=2,default=[100,200])
    vis_parser.add_argument("--annotate_color",type=int,nargs=3,default=[0,0,255])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    save_name_list = [re.search(args.save_pattern, img_name).group() for img_name in args.img_path]
    save_name_list = ["{}_{}.png".format(method, name) for method, name in zip(args.method, save_name_list)]
    marker = cv2.imread(args.marker_path, cv2.IMREAD_GRAYSCALE)
    marker_area = cv2.Canny(marker, args.canny_threshold[0], args.canny_threshold[1])
    os.makedirs(args.save_dir, exist_ok=True)
    for img_name, save_name in zip(args.img_path, save_name_list):
        img = cv2.imread(img_name)
        img[marker_area > 0,:] = np.array(args.annotate_color, dtype=np.int32)
        cv2.imwrite(os.path.join(args.save_dir, save_name), img)
        