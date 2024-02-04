from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import normalized_mutual_information as NMI
from skimage.metrics import mean_squared_error as MSE
from cleanfid import fid
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir",type=str,default="dataset/taxim/asim_non_marker")
    parser.add_argument("--pred_dir",type=str,default="results/taxim_m1/MAE")
    parser.add_argument("--log_file",type=str,default="log/taxim_m1.txt")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    index_dict = dict()
    print("Compute FID...")
    fid_idx = fid.compute_fid(args.gt_dir, args.pred_dir)
    print("Compute FID...")
    kid_idx = fid.compute_kid(args.gt_dir, args.pred_dir)
    gt_img_names = list(sorted(os.listdir(args.gt_dir)))
    pred_img_names = list(sorted(os.listdir(args.pred_dir)))
    ssim_list = []
    psnr_list = []
    mse_list = []
    for gt_name, pred_name in tqdm(zip(gt_img_names, pred_img_names),total=len(gt_img_names)):
        gt_image = cv2.imread(os.path.join(args.gt_dir, gt_name),cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(os.path.join(args.pred_dir, pred_name), cv2.IMREAD_GRAYSCALE)
        ssim_list.append(SSIM(gt_image, pred_image))
        psnr_list.append(PSNR(gt_image, pred_image))
        # nmi_list.append(NMI(gt_image, pred_image))
        mse_list.append(MSE(gt_image, pred_image))
    ssim = sum(ssim_list) / len(ssim_list)
    psnr = sum(psnr_list) / len(psnr_list)
    # nmi = sum(nmi_list) / len(nmi_list)
    mse = sum(mse_list) / len(mse_list)
    with open(args.log_file,'a') as f:
        f.write("GT Dir: {}\n".format(args.gt_dir))
        f.write("Pred Dir: {}\n".format(args.pred_dir))
        f.write("FID: {}\n".format(fid_idx))
        f.write("KID: {}\n".format(kid_idx))
        f.write("MSE: {}\n".format(mse))
        f.write("SSIM: {}\n".format(ssim))
        f.write("PSNR: {}\n".format(psnr))