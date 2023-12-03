import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error as MSE

def rmse(gt:np.ndarray, pred:np.ndarray):
    return np.sqrt(np.sum((gt-pred)**2, axis=-1))

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",type=int,default=1)
    parser.add_argument("--gt_path",type=str,default="dataset/taxim/asim_non_marker_256")
    parser.add_argument("--pred_fmt",type=str,default="results/taxim_m1_256/{method}")
    parser.add_argument("--method_list",type=str,nargs="+",default=["NS","TELEA","Palette"])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    method_dirs = [args.pred_fmt.format(method=m) for m in args.method_list]
    gt_img_file = sorted(os.listdir(args.gt_path))[args.index]
    pred_img_files = [sorted(os.listdir(method_dir))[args.index] for method_dir in method_dirs]
    gt_img = cv2.cvtColor(cv2.imread(os.path.join(args.gt_path, gt_img_file)), cv2.COLOR_RGB2BGR)
    pred_imgs = [cv2.cvtColor(cv2.imread(os.path.join(method_dir, pred_img_file)), cv2.COLOR_RGB2BGR)  for method_dir, pred_img_file in zip(method_dirs,pred_img_files)]
    maes = [rmse(gt_img, pred_img) for pred_img in pred_imgs]
    rmses = [MSE(gt_img, pred_img) for pred_img in pred_imgs]
    max_rmse = rmses[0].max()
    for i in range(1, len(rmses)):
        max_rmse = max(max_rmse, rmses[i].max())
    for i in range(len(pred_imgs)):
        plt.subplot(2,len(pred_imgs),i+1)
        plt.imshow(pred_imgs[i])
        plt.title(args.method_list[i] + " : {:.2f}".format(rmses[i]))
    for i in range(len(maes)):
        plt.subplot(2,len(pred_imgs),i+len(pred_imgs)+1)
        plt.imshow(maes[i], vmax = max_rmse, vmin=0)
    plt.tight_layout()
    plt.savefig("demo_rmse.png")