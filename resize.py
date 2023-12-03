import argparse
import cv2
import os
from tqdm import tqdm
import shutil

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--src_img_dir",type=str,default="dataset/test/rand2")
    io_parser.add_argument("--src_marker_dir",type=str,default="dataset/test/rand_marker2")
    io_parser.add_argument("--tgt_img_dir",type=str,default="dataset/test/rand2_256")
    io_parser.add_argument("--tgt_marker_dir",type=str,default="dataset/test/marker2_256")
    io_parser.add_argument("--tgt_wmarker_dir",type=str,default="dataset/test/wmarker2_256")
    io_parser.add_argument("--output_tgt_img",type=bool,default=True)
    para_parser = parser.add_argument_group()
    para_parser.add_argument("--target_size",type=int,nargs=2,default=[256,256])
    return parser.parse_args()

def refresh_dir(dirname:str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

def renew_suffix(filename:str, suffix:str='.bmp'):
    return os.path.splitext(filename)[0] + suffix

if __name__ == "__main__":
    args = options()
    src_img_list = list(sorted(os.listdir(args.src_img_dir)))
    src_marker_list = list(sorted(os.listdir(args.src_marker_dir)))
    if args.output_tgt_img:
        refresh_dir(args.tgt_img_dir)
    refresh_dir(args.tgt_marker_dir)
    refresh_dir(args.tgt_wmarker_dir)
    for src_imgname, src_markername in tqdm(zip(src_img_list, src_marker_list),total=len(src_img_list)):
        src_img = cv2.imread(os.path.join(args.src_img_dir, src_imgname))
        src_marker_img = cv2.imread(os.path.join(args.src_marker_dir, src_markername), cv2.IMREAD_GRAYSCALE)
        tgt_img = cv2.resize(src_img, args.target_size, interpolation=cv2.INTER_AREA)
        tgt_marker_img = cv2.resize(src_marker_img, args.target_size, interpolation=cv2.INTER_AREA)
        tgt_marker_img[tgt_marker_img>0] = 255
        overlay_img = tgt_img.copy()
        overlay_img[tgt_marker_img > 0] = 0
        if args.output_tgt_img:
            cv2.imwrite(os.path.join(args.tgt_img_dir, renew_suffix(src_imgname)), tgt_img)
        cv2.imwrite(os.path.join(args.tgt_marker_dir, renew_suffix(src_markername)), tgt_marker_img)
        cv2.imwrite(os.path.join(args.tgt_wmarker_dir, renew_suffix(src_markername)), overlay_img)