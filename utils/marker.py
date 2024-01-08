import cv2
import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
# import sys
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join("..",os.path.dirname(__file__)))
# from .utils import refresh_dir

def bin2uint8(img:np.ndarray):
    return np.array(255 * img, dtype=np.uint8)

def find_marker(frame,
        morphop_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        mask_range=(150, 255), min_value:int=70,
        morphop_iter=1, morphclose_iter=2, dilate_iter=1):
    """find markers in the tactile iamge

    Args:
        frame (np.ndarray): input image (can be RGB or grayscale)
        morphop_kernel (cv.MatLike, optional): kernel of MORPH_OPEN. Defaults to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)).
        morphclose_kernel (cv.MatLike, optional): kernel of MORPH_CLOSE. Defaults to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)).
        dilate_kernel (cv.MatLike, optional): kernel of dilation. Defaults to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)).
        mask_range (tuple, optional): range of mask segementation. Defaults to (150, 255).
        min_value (int, optional): minimum value to segement marker from HSV (V-chan). Defaults to 70.
        morphop_iter (int, optional): iteration of MORPH_OPEN operation. Defaults to 1.
        morphclose_iter (int, optional): iteration of MORPH_CLOSE operation. Defaults to 2.
        dilate_iter (int, optional): iteration of DILATION operation. Defaults to 1.

    Returns:
        np.ndarray: final mask (0, 255) np.unint8
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) ### use only the green channel
    value = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[...,-1]
    # img_sblur = cv2.GaussianBlur(gray,(3,3),5)
    img_lblur = cv2.GaussianBlur(gray, (15,15),5)
    im_blur_sub = img_lblur - gray + 128
    blur_mask = np.logical_and(im_blur_sub >= mask_range[0], im_blur_sub <= mask_range[1])
    value_mask = value < min_value
    mask = np.logical_or(blur_mask, value_mask)
    mask255 = np.array(255 * mask,dtype=np.uint8)
    mask255_op = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, morphop_kernel, iterations=morphop_iter)
    dilate_mask = cv2.dilate(mask255_op, dilate_kernel, iterations=dilate_iter)
    morph_close = cv2.morphologyEx(dilate_mask, cv2.MORPH_CLOSE, morphclose_kernel, iterations=morphclose_iter)
    return morph_close


def find_marker_debug(frame, mask_range=(145, 255), min_value:int=70, morphop_size=5, morphop_iter=1, dilate_size=5, dilate_iter=1):
    """find markers in the tactile iamge

    Args:
        frame (`np.ndarray`): raw image
        mask_range (tuple, optional): range of guassian difference. Defaults to (150, 255).
        min_value (int, optional): minimum value to segment markers. Defaults to 60.
        dilate_size (int, optional): size of dilation. Defaults to 60.
        dilate_iter (int, optional): iterations of dilation. Defaults to 1.
    Returns:
        `np.ndarray`: 0,255 np.uint8
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) ### use only the green channel
    value = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[...,-1]
    # img_sblur = cv2.GaussianBlur(gray,(3,3),5)
    img_lblur = cv2.GaussianBlur(gray, (15,15),5)
    im_blur_sub = img_lblur - gray + 128
    blur_mask = np.logical_and(im_blur_sub >= mask_range[0], im_blur_sub <= mask_range[1])
    value_mask = value < min_value
    mask = np.logical_or(blur_mask, value_mask)
    mask255 = np.array(255 * mask,dtype=np.uint8)
    morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphop_size, morphop_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    mask255_op = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, morph_open_kernel, iterations=morphop_iter)
    dilate_mask = cv2.dilate(mask255_op, dilate_kernel, iterations=dilate_iter)
    morph_close = cv2.morphologyEx(dilate_mask, cv2.MORPH_CLOSE, dilate_kernel, iterations=2)
    return gray, value, img_lblur, im_blur_sub, blur_mask, value_mask, mask255, mask255_op, dilate_mask, morph_close

def detect_empty_img(frame:np.ndarray, mask_range=(145, 255), blur_ksize=(15, 15),min_value=70, morphop_size=5, morphop_iter=1, empty_threshold=0.005):
    """detect empty tactile image without any indention

    Args:
        frame (np.ndarray): _description_
        mask_range (tuple, optional): _description_. Defaults to (145, 255).
        blur_ksize (tuple, optional): _description_. Defaults to (15, 15).
        morphop_size (int, optional): _description_. Defaults to 5.
        morphop_iter (int, optional): _description_. Defaults to 1.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) ### use only the green channel
    value = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[...,-1]
    value_mask = value < min_value
    # img_sblur = cv2.GaussianBlur(gray,(3,3),5)
    img_lblur = cv2.GaussianBlur(gray, blur_ksize,5)
    im_blur_sub = img_lblur - gray + 128
    blur_mask = np.logical_and(im_blur_sub >= mask_range[0], im_blur_sub <= mask_range[1])
    mask = np.logical_or(blur_mask, value_mask)
    mask255 = np.array(255 * mask,dtype=np.uint8)
    morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphop_size, morphop_size))
    mask255_op = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, morph_open_kernel, iterations=morphop_iter)
    return np.sum(mask255_op != mask255)/mask255.size < empty_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",type=str,default="dataset/vitac/img/000928.jpg")
    parser.add_argument("--output_dir",type=str,default="debug_mask")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    raw_img = cv2.imread(args.img)
    gray, value, img_lblur, im_blur_sub, blur_mask, value_mask, mask255, mask255_op, dilate_mask, morph_close = find_marker_debug(raw_img)
    cv2.imwrite(os.path.join(args.output_dir, 'raw.png'), raw_img)
    cv2.imwrite(os.path.join(args.output_dir, 'gray.png'), gray)
    cv2.imwrite(os.path.join(args.output_dir, 'value.png'), value)
    # cv2.imwrite(os.path.join(args.output_dir, 'img_sblur.png'), img_sblur)
    cv2.imwrite(os.path.join(args.output_dir, 'img_lblur.png'), img_lblur)
    plt.figure()
    plt.imshow(im_blur_sub, cmap=get_cmap("gist_gray"))
    plt.axis("off")
    plt.tight_layout(pad=0,h_pad=0,w_pad=0)
    plt.savefig(os.path.join(args.output_dir, 'im_blur_sub.png'))
    cv2.imwrite(os.path.join(args.output_dir, 'blur_mask.png'), bin2uint8(blur_mask))
    cv2.imwrite(os.path.join(args.output_dir, 'value_mask.png'), bin2uint8(value_mask))
    cv2.imwrite(os.path.join(args.output_dir, 'mask.png'), mask255)
    cv2.imwrite(os.path.join(args.output_dir, 'filtered.png'), mask255_op)
    cv2.imwrite(os.path.join(args.output_dir, 'dilate_mask.png'), dilate_mask)
    cv2.imwrite(os.path.join(args.output_dir, 'closed_mask.png'), morph_close)
    print(np.sum(mask255_op != mask255)/mask255.size)
    print(detect_empty_img(raw_img))