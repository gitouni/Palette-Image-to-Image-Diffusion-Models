import argparse
import os
import cv2
from tqdm import tqdm
import shutil

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_continue",type=bool,default=False)
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--input_dir",type=str,default="dataset/slip_sp/markered")
    io_parser.add_argument("--marker_dir",type=str,default="dataset/slip_sp/marker")
    io_parser.add_argument("--inpaint_dir",type=str,default="dataset/slip_sp/TELEA")
    run_parser = parser.add_argument_group()
    run_parser.add_argument("--dilate",type=bool,default=True)
    run_parser.add_argument("--kernel_size",type=int,default=3)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    img_files = list(sorted(os.listdir(args.input_dir)))
    marker_files = list(sorted(os.listdir(args.marker_dir)))
    if os.path.exists(args.inpaint_dir):
        shutil.rmtree(args.inpaint_dir)
    os.makedirs(args.inpaint_dir)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    for img_file, marker_file in tqdm(zip(img_files, marker_files),total=len(img_files)):
        img = cv2.imread(os.path.join(args.input_dir, img_file))
        marker_img = cv2.imread(os.path.join(args.marker_dir, marker_file), cv2.IMREAD_GRAYSCALE)
        if args.dilate:
            marker_img = cv2.dilate(marker_img, kernel)
        H, W = img.shape[:2]
        inpaint_img = cv2.inpaint(img, marker_img,args.kernel_size,cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(args.inpaint_dir, os.path.splitext(img_file)[0]+".png"), inpaint_img)
        