from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import normalized_mutual_information as NMI
from skimage.metrics import mean_squared_error as MSE
from torch_fidelity import calculate_metrics
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from data.dataset import PatchImageDataset

def MAE(src:np.ndarray, tgt:np.ndarray):
    return np.mean(np.abs(src.astype(np.float64)-tgt.astype(np.float64)))

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir",type=str,default="dataset/marker_adding/markerless")
    parser.add_argument("--pred_dir",type=str,default="dataset/marker_adding/Palette")
    parser.add_argument("--log_file",type=str,default="log/marker_adding.txt")
    parser.add_argument("--fid_winsize",type=int,nargs=2,default=[128,128])
    parser.add_argument("--overlap",type=int,nargs=2,default=[0,0])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    index_dict = dict()
    gt_img_names = list(sorted(os.listdir(args.gt_dir)))
    pred_img_names = list(sorted(os.listdir(args.pred_dir)))
    gt_img_dataset = PatchImageDataset(args.fid_winsize, overlap=args.overlap, buffer_size=5, data_root=args.gt_dir)
    pred_img_dataset = PatchImageDataset(args.fid_winsize, overlap=args.overlap, buffer_size=5, data_root=args.pred_dir)
    print("Computing FID...")
    fid_metrics = calculate_metrics(input1=gt_img_dataset, input2=pred_img_dataset, cuda=True, fid=True, kid=True)
    fid = fid_metrics['frechet_inception_distance']
    kid = fid_metrics['kernel_inception_distance_mean']
    ssim_list = []
    psnr_list = []
    nmi_list = []
    mse_list = []
    mae_list = []
    for gt_name, pred_name in tqdm(zip(gt_img_names, pred_img_names),total=len(gt_img_names)):
        gt_image = cv2.imread(os.path.join(args.gt_dir, gt_name),cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(os.path.join(args.pred_dir, pred_name), cv2.IMREAD_GRAYSCALE)
        ssim_list.append(SSIM(gt_image, pred_image))
        psnr_list.append(PSNR(gt_image, pred_image))
        nmi_list.append(NMI(gt_image, pred_image))
        mse_list.append(MSE(gt_image, pred_image))
        mae_list.append(MAE(gt_image, pred_image))
    ssim = sum(ssim_list) / len(ssim_list)
    psnr = sum(psnr_list) / len(psnr_list)
    nmi = sum(nmi_list) / len(nmi_list)
    mse = sum(mse_list) / len(mse_list)
    mae = sum(mae_list) / len(mae_list)
    with open(args.log_file,'a') as f:
        f.write("GT Dir: {}\n".format(args.gt_dir))
        f.write("Pred Dir: {}\n".format(args.pred_dir))
        f.write("FID: {}\n".format(fid))
        f.write("KID: {}\n".format(kid))
        f.write("NMI: {}\n".format(nmi))
        f.write("MSE: {}\n".format(mse))
        f.write("MAE: {}\n".format(mae))
        f.write("SSIM: {}\n".format(ssim))
        f.write("PSNR: {}\n".format(psnr))