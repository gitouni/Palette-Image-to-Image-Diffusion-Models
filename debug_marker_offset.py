import cv2
import numpy as np
import argparse
import os
import shutil

def refresh_dir(dirname:str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--raw_input",type=str,default="results/taxim_m1/Palette/Out_0001.bmp")
    io_parser.add_argument("--raw_mask",type=str,default="dataset/taxim/marker/0001.png")
    io_parser.add_argument("--output_root",type=str,default="debug/mask_offset/")
    io_parser.add_argument("--cropped_ant_raw",type=str,default="cropped_ant_raw.png")
    io_parser.add_argument("--cropped_ant_marker",type=str,default="cropped_ant_marker.png")
    io_parser.add_argument("--cropped_img",type=str,default="cropped_raw.png")
    io_parser.add_argument("--cropped_mask",type=str,default="cropped_mask.png")
    io_parser.add_argument("--new_mask",type=str,default="new_mask.png")
    io_parser.add_argument("--new_with_marker",type=str,default="wmarker.png")
    para_parser = parser.add_argument_group()
    para_parser.add_argument("--offset",type=int,nargs=2,default=[20,20])
    para_parser.add_argument("--cropped_size",type=int,default=[256,256], help='height, width')
    para_parser.add_argument("--cropped_thickness",type=int,default=2)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    raw_input = cv2.imread(args.raw_input)
    raw_mask = cv2.imread(args.raw_mask, cv2.IMREAD_GRAYSCALE)
    refresh_dir(args.output_root)
    H, W = raw_mask.shape
    mask_yx = np.nonzero(raw_mask)  # y,x
    new_mask_yx = [mask_yx[0]+args.offset[0], mask_yx[1]+args.offset[1]]
    rev = (new_mask_yx[0] >= 0) * (new_mask_yx[0] < H) * (new_mask_yx[1] >= 0) * (new_mask_yx[1] < W)
    new_mask_yx = [new_mask_yx[0][rev], new_mask_yx[1][rev]]
    new_mask = np.zeros_like(raw_mask, dtype=np.uint8)
    new_mask[new_mask_yx[0], new_mask_yx[1]] = 255
    cv2.imwrite(os.path.join(args.output_root, args.new_mask),new_mask)
    new_output = np.copy(raw_input)
    new_output[new_mask > 0] = 0
    cv2.imwrite(os.path.join(args.output_root, args.new_with_marker),new_output)
    cropped_x0, cropped_y0 = np.random.randint(low=0,high=W-args.cropped_size[1]), np.random.randint(low=0,high=H-args.cropped_size[0])
    cropped_img = raw_input[cropped_y0:cropped_y0+args.cropped_size[0], cropped_x0:cropped_x0+args.cropped_size[1]]
    cv2.imwrite(os.path.join(args.output_root,args.cropped_img), cropped_img)
    cropped_ant = cv2.rectangle(raw_input, (cropped_x0, cropped_y0), (cropped_x0 + args.cropped_size[1], cropped_y0 + args.cropped_size[0]), color=(255,255,255), thickness=args.cropped_thickness)
    cv2.imwrite(os.path.join(args.output_root,args.cropped_ant_raw), cropped_ant)
    cropped_mask = new_mask[cropped_y0:cropped_y0+args.cropped_size[0], cropped_x0:cropped_x0+args.cropped_size[1]]
    cv2.imwrite(os.path.join(args.output_root,args.cropped_mask), cropped_mask)
    cropped_ant_marker = cv2.rectangle(new_mask, (cropped_x0, cropped_y0), (cropped_x0 + args.cropped_size[1], cropped_y0 + args.cropped_size[0]), color=255, thickness=args.cropped_thickness)
    cv2.imwrite(os.path.join(args.output_root,args.cropped_ant_marker), cropped_ant_marker)
    