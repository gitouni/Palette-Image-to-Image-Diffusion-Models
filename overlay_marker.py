import cv2
import os
import argparse
import numpy as np
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",type=str,default="dataset/train_val/rand_img")
    parser.add_argument("--marker_dir",type=str,default="dataset/train_val/rand_marker2")
    parser.add_argument("--output_dir",type=str,default="dataset/train_val/rand_wmarker2")
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    img_file_list = sorted(os.listdir(args.img_dir))
    marker_file_list = sorted(os.listdir(args.marker_dir))
    for img_file, marker_file in zip(img_file_list, marker_file_list):
        img = cv2.imread(os.path.join(args.img_dir, img_file))
        marker = cv2.imread(os.path.join(args.marker_dir, marker_file), cv2.IMREAD_GRAYSCALE)
        img[marker > 0, :] = np.array([0,0,0],dtype=np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, os.path.splitext(img_file)[0]+'.bmp'), img)