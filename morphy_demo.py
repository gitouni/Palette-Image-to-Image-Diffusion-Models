import cv2
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",type=int,nargs=2,default=[50,50])
    parser.add_argument("--radius",type=int,default=15)
    parser.add_argument("--noise_num",type=int,default=50)
    parser.add_argument("--output_dir",type=str,default="debug_morp")
    parser.add_argument("--dilate_size",type=int,default=3)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    h, w = args.size
    r = args.radius
    img = np.zeros([h, w],dtype=np.uint8)
    img = cv2.circle(img, center=(h//2,w//2), radius=r, color=255, thickness=-1)
    cv2.imwrite(os.path.join(args.output_dir,'raw.png'),img)
    noise_dots_x = np.array(np.random.rand(args.noise_num) * w, dtype=np.int32)
    noise_dots_y = np.array(np.random.rand(args.noise_num) * h, dtype=np.int32)
    noise_color = img[noise_dots_y, noise_dots_x]
    img[noise_dots_y, noise_dots_x] = 255 - noise_color
    cv2.imwrite(os.path.join(args.output_dir,'noised.png'),img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(args.dilate_size, args.dilate_size))
    open_stage1 = cv2.erode(img, kernel)
    open_stage2 = cv2.dilate(open_stage1, kernel)
    close_stage1 = cv2.dilate(img, kernel)
    close_stage2 = cv2.erode(close_stage1, kernel)
    cv2.imwrite(os.path.join(args.output_dir, "open_s1.png"), open_stage1)
    cv2.imwrite(os.path.join(args.output_dir, "open_s2.png"), open_stage2)
    cv2.imwrite(os.path.join(args.output_dir, "close_s1.png"), close_stage1)
    cv2.imwrite(os.path.join(args.output_dir, "close_s2.png"), close_stage2)