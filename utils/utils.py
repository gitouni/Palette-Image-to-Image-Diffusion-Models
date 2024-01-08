"""
poisson_reconstruct.py
Fast Poisson Reconstruction in Python
Copyright (c) 2014 Jack Doerner
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
import numpy
import scipy, scipy.fftpack
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy import ndimage
from scipy.signal import fftconvolve

import os
import shutil

def refresh_dir(dirname:str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)
    
def poisson_reconstruct(grady, gradx, boundarysrc):
	# Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
	# Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

	# Laplacian
	gyy = grady[1:,:-1] - grady[:-1,:-1]
	gxx = gradx[:-1,1:] - gradx[:-1,:-1]
	f = numpy.zeros(boundarysrc.shape)
	f[:-1,1:] += gxx
	f[1:,:-1] += gyy

	# Boundary image
	boundary = boundarysrc.copy()
	boundary[1:-1,1:-1] = 0

	# Subtract boundary contribution
	f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
	f = f[1:-1,1:-1] - f_bp

	# Discrete Sine Transform
	tt = scipy.fftpack.dst(f, norm='ortho')
	fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

	# Eigenvalues
	(x,y) = numpy.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
	denom = (2*numpy.cos(math.pi*x/(f.shape[1]+2))-2) + (2*numpy.cos(math.pi*y/(f.shape[0]+2)) - 2)

	f = fsin/denom

	# Inverse Discrete Sine Transform
	tt = scipy.fftpack.idst(f, norm='ortho')
	img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

	# New center + old boundary
	result = boundary
	result[1:-1,1:-1] = img_tt

	return result






def gkern(l=5, sig=1.):
    """ creates gaussian kernel with side length l and a sigma of sig """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)  # normalize

def normxcorr2(template, image, mode="same"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")
    template = template - np.mean(template)
    image = image - np.mean(image)
    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))
    # Remove small machine precision errors after subtraction
    image[image < 0] = 0
    template = np.sum(np.square(template))
    den = np.sqrt(image * template)
    valid_mask = den != 0
    out[valid_mask] = out[valid_mask] / den[valid_mask]
    # Remove any divisions by 0 or very close to 0
    out[~valid_mask] = 0
    return out


def find_marker(frame, mask_range=(144, 255), gkern_length=5, gkern_sig=1.5):
    """find markers in the tactile iamge

    Args:
        frame (`np.ndarray`): raw image
        mask_range (tuple, optional): range of guassian difference. Defaults to (144, 255).
        gkern_length (int, optional): length of the guassian kernel. Defaults to 5.
        gkern_sig (float, optional): std of the guassian kernel. Defaults to 1.5.

    Returns:
        `np.ndarray`: 0,1 np.uint8
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) ### use only the green channel
    im_blur_3 = cv2.GaussianBlur(gray,(3,3),5)
    im_blur_8 = cv2.GaussianBlur(gray, (15,15),5)
    im_blur_sub = im_blur_8 - im_blur_3 + 128
    mask = cv2.inRange(im_blur_sub, mask_range[0], mask_range[1])

    # ''' normalized cross correlation '''
    template = gkern(l=gkern_length, sig=gkern_sig)
    nrmcrimg = normxcorr2(template, mask)
    # ''''''''''''''''''''''''''''''''''''
    a = nrmcrimg
    mask = np.asarray(a > 0.1)
    mask = (mask).astype('uint8')

    return mask

def find_marker2(frame, mask_range=(144, 255), dilate_size=5, dilate_iter=1, k1=(3,3),k2=(15,15)):
    """find markers in the tactile iamge

    Args:
        frame (`np.ndarray`): raw image
        mask_range (tuple, optional): range of guassian difference. Defaults to (144, 255).
        dilate_size (int, optional): size of dilation kernel. Defaults to 5.
        dilate_iter (float, optional): iteration of dilation. Defaults to 1.5.

    Returns:
        `np.ndarray`: 0,1 np.uint8
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) ### use only the green channel
    im_blur_3 = cv2.GaussianBlur(gray,k1,5)
    im_blur_8 = cv2.GaussianBlur(gray,k2,5)
    im_blur_sub = im_blur_8 - im_blur_3 + 128
    blur_mask = cv2.inRange(im_blur_sub, mask_range[0], mask_range[1])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(dilate_size, dilate_size))
    mask = cv2.dilate(blur_mask, kernel, iterations=dilate_iter)
    return mask


def find_marker_debug(frame, mask_range=(144, 255), gkern_length=5, gkern_sig=1.5):
    """find markers in the tactile iamge

    Args:
        frame (`np.ndarray`): raw image
        mask_range (tuple, optional): range of guassian difference. Defaults to (144, 255).
        gkern_length (int, optional): length of the guassian kernel. Defaults to 5.
        gkern_sig (float, optional): std of the guassian kernel. Defaults to 1.5.

    Returns:
        `np.ndarray`: 0,1 np.uint8
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) ### use only the green channel
    im_blur_3 = cv2.GaussianBlur(gray,(3,3),5)
    im_blur_8 = cv2.GaussianBlur(gray, (15,15),5)
    im_blur_sub = im_blur_8 - im_blur_3 + 128
    blur_mask = cv2.inRange(im_blur_sub, mask_range[0], mask_range[1])

    # ''' normalized cross correlation '''
    template = gkern(l=gkern_length, sig=gkern_sig)
    nrmcrimg = normxcorr2(template, blur_mask)
    # ''''''''''''''''''''''''''''''''''''
    mask = np.asarray(nrmcrimg > 0.1)
    mask = (mask).astype('uint8')

    return gray, im_blur_3, im_blur_8, im_blur_sub, blur_mask, template, nrmcrimg, mask


def find_marker_debug2(frame, mask_range=(144, 255), dilate_size=5, dilate_iter=1, k1=(3,3), k2=(15,15)):
    """find markers in the tactile iamge

    Args:
        frame (`np.ndarray`): raw image
        mask_range (tuple, optional): range of guassian difference. Defaults to (144, 255).
        dilate_size (int, optional): size of dilation kernel. Defaults to 5.
        dilate_iter (float, optional): iteration of dilation. Defaults to 1.5.

    Returns:
        `np.ndarray`: 0,1 np.uint8
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) ### use only the green channel
    value = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[...,-1]
    im_blur_3 = cv2.GaussianBlur(gray,k1,5)
    im_blur_8 = cv2.GaussianBlur(gray, k2,5)
    im_blur_sub = im_blur_8 - im_blur_3 + 128
    blur_mask = cv2.inRange(im_blur_sub, mask_range[0], mask_range[1])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(dilate_size, dilate_size))
    value_mask = value < 70
    mask = np.array(255*np.logical_and(blur_mask, value_mask), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return gray, im_blur_3, im_blur_8, im_blur_sub, blur_mask, mask, value, value_mask

def marker_center(mask, frame):

    ''' first method '''
    # RESCALE = setting.RESCALE
    # areaThresh1=30/RESCALE**2
    # areaThresh2=1920/RESCALE**2
    # MarkerCenter = []
    # contours=cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours[1])<25:  # if too little markers, then give up
    #     print("Too less markers detected: ", len(contours))
    #     return MarkerCenter
    # for contour in contours[1]:
    #     x,y,w,h = cv2.boundingRect(contour)
    #     AreaCount=cv2.contourArea(contour)
    #     # print(AreaCount)
    #     if AreaCount>areaThresh1 and AreaCount<areaThresh2 and abs(np.max([w, h]) * 1.0 / np.min([w, h]) - 1) < 1:
    #         t=cv2.moments(contour)
    #         # print("moments", t)
    #         # MarkerCenter=np.append(MarkerCenter,[[t['m10']/t['m00'], t['m01']/t['m00'], AreaCount]],axis=0)
    #         mc = [t['m10']/t['m00'], t['m01']/t['m00']]
    #         # if t['mu11'] < -100: continue
    #         MarkerCenter.append(mc)
    #         # print(mc)
    #         cv2.circle(frame, (int(mc[0]), int(mc[1])), 10, ( 0, 0, 255 ), 2, 6);

    ''' second method '''
    img3 = mask
    neighborhood_size = 10
    # threshold = 40 # for r1.5
    threshold = 0 # for mini
    data_max = maximum_filter(img3, neighborhood_size)
    maxima = (img3 == data_max)
    data_min = minimum_filter(img3, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    MarkerCenter = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))
    MarkerCenter[:, [0, 1]] = MarkerCenter[:, [1, 0]]
    for i in range(MarkerCenter.shape[0]):
        x0, y0 = int(MarkerCenter[i][0]), int(MarkerCenter[i][1])
        cv2.circle(mask, (x0, y0), color=(0, 0, 0), radius=1, thickness=1)
    return MarkerCenter

def draw_flow(frame, flow):
    Ox, Oy, Cx, Cy, Occupied = flow

    dx = np.mean(np.abs(np.asarray(Ox) - np.asarray(Cx)))
    dy = np.mean(np.abs(np.asarray(Oy) - np.asarray(Cy)))
    dnet = np.sqrt(dx**2 + dy**2)
    print (dnet * 0.075, '\n')


    K = 1
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
            color = (0, 0, 255)
            if Occupied[i][j] <= -1:
                color = (127, 127, 255)
            cv2.arrowedLine(frame, pt1, pt2, color, 2,  tipLength=0.25)


def warp_perspective(img):

    TOPLEFT = (175,230)
    TOPRIGHT = (380,225)
    BOTTOMLEFT = (10,410)
    BOTTOMRIGHT = (530,400)

    WARP_W = 215
    WARP_H = 215

    points1=np.float32([TOPLEFT,TOPRIGHT,BOTTOMLEFT,BOTTOMRIGHT])
    points2=np.float32([[0,0],[WARP_W,0],[0,WARP_H],[WARP_W,WARP_H]])

    matrix=cv2.getPerspectiveTransform(points1,points2)

    result = cv2.warpPerspective(img, matrix, (WARP_W,WARP_H))

    return result


def init_HSR(img):
    DIM=(640, 480)
    img = cv2.resize(img, DIM)

    K=np.array([[225.57469247811056, 0.0, 280.0069549918857], [0.0, 221.40607131318117, 294.82435570493794], [0.0, 0.0, 1.0]])
    D=np.array([[0.7302503082668154], [-0.18910060205317372], [-0.23997727800712282], [0.13938490908400802]])
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return warp_perspective(undistorted_img)