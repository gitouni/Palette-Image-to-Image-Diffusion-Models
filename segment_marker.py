import argparse
import os
from functools import partial
import cv2
from tqdm import tqdm
import numpy as np

def find_marker(frame, kernel, dilate_iter, value_threshold=70):
    """find markers in the tactile iamge

    Args:
        frame (`np.ndarray`): raw image
        mask_range (tuple, optional): range of guassian difference. Defaults to (144, 255).
        dilate_size (int, optional): size of dilation kernel. Defaults to 5.
        dilate_iter (float, optional): iteration of dilation. Defaults to 1.5.

    Returns:
        `np.ndarray`: 0,1 np.uint8
    """
    value = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[...,-1]
    value_mask = np.array(255*(value < value_threshold),dtype=np.uint8)
    mask = cv2.dilate(value_mask, kernel, iterations=dilate_iter)
    return mask

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--input_dir",type=str,default="dataset/vitac/img")
    io_parser.add_argument("--output_dir",type=str,default="dataset/vitac/marker")
    marker_parser = parser.add_argument_group()
    marker_parser.add_argument("--dilate_size",type=int,default=3)
    marker_parser.add_argument("--dilate_iter",type=int,default=1)
    marker_parser.add_argument("--value_threshold",type=int,default=60)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    os.makedirs(args.output_dir, exist_ok=True)
    calib_find_marker = partial(find_marker, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(args.dilate_size, args.dilate_size)), dilate_iter=args.dilate_iter, value_threshold=args.value_threshold)
    img_list = list(sorted(os.listdir(args.input_dir)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    for img_name in tqdm(img_list):
        img = cv2.imread(os.path.join(args.input_dir, img_name))
        marker_mask = calib_find_marker(img)
        marker_mask = cv2.dilate(marker_mask, kernel)
        cv2.imwrite(os.path.join(args.output_dir, os.path.splitext(img_name)[0]+".png"),marker_mask)