###################################################
### PREPROCESSING OF WSIs (SVS format) TO PATCHES
###################################################
### This script extracts smaller non-overlapping patches from a high-dimensional WSI
### - step 1: OTSU segmentation is performed to extract target tissue from the background
### - step 2: Extraction of non-overlapping patches from foreground at 20x resolution using OpenSlide library
###################################################
###################################################
### Example command
### $ python 1_WSI2Patches.py --wsi_patch "/path/to/WSIfolder/" --patch_path "/path/to/output/patchfolder/" --mask_path "/path/to/output/maskfolder/" 
###                       --patch_size 224 --max_patches_per_slide 2000 --num_process 10 --dezoom_factor 1.0
###################################################
###################################################

### Set Environment
####################

import pandas as pd
import numpy as np
from openslide import OpenSlide
from multiprocessing import Pool, Value, Lock
import os
from IPython.display import display
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.exposure.exposure import is_low_contrast
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import argparse
import logging

### Functions
###############

def get_mask_image(img_RGB, RGB_min=50):
    
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

def get_mask(slide, level='max', RGB_min=50):
    
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level

def extract_patches(slide_path, mask_path, patch_size, patches_output_dir, slide_id, max_patches_per_slide=2000):
    
    patch_folder = os.path.join(patches_output_dir, slide_id)
    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)
    slide = OpenSlide(slide_path)

    patch_folder_mask = os.path.join(mask_path, slide_id)
    if not os.path.isdir(patch_folder_mask):
        os.makedirs(patch_folder_mask)
        mask, mask_level = get_mask(slide)
        mask = binary_dilation(mask, iterations=3)
        mask = binary_erosion(mask, iterations=3)
        np.save(os.path.join(patch_folder_mask, "mask.npy"), mask) 
    else:
        mask = np.load(os.path.join(mask_path, slide_id, 'mask.npy'))
        
    mask_level = len(slide.level_dimensions) - 1
    
    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    try:
        with open(os.path.join(patch_folder, 'loc.txt'), 'w') as loc:
            loc.write("slide_id {0}\n".format(slide_id))
            loc.write("id x y patch_level patch_size_read patch_size_output\n")

            ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
            ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

            xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

            # handle slides with 40 magnification at base level
            resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
            resize_factor = resize_factor * args.dezoom_factor
            patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
            i = 0

            indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                       range(0, ymax, patch_size_resized[0])]
            np.random.seed(5)
            np.random.shuffle(indices)
            for x, y in indices:
                # check if in background mask
                x_mask = int(x / ratio_x)
                y_mask = int(y / ratio_y)
                if mask[x_mask, y_mask] == 1:
                    patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                    try:
                        mask_patch = get_mask_image(np.array(patch))
                        mask_patch = binary_dilation(mask_patch, iterations=3)
                    except Exception as e:
                        print("error with slide id {} patch {}".format(slide_id, i))
                        print(e)
                    if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                        if resize_factor != 1.0:
                            patch = patch.resize(patch_size)
                        loc.write("{0} {1} {2} {3} {4} {5}\n".format(i, x, y, PATCH_LEVEL, patch_size_resized[0],
                                                                     patch_size_resized[1]))
                        imsave(os.path.join(patch_folder, "{0}_patch_{1}.png".format(slide_id, i)), np.array(patch))
                        i += 1
                if i >= max_patches_per_slide:
                    break

            if i == 0:
                print("no patch extracted for slide {}".format(slide_id))
    except Exception as e:
        print("error with slide id {} patch {}".format(slide_id, i))
        print(e)

def get_slide_id(slide_name):
    return slide_name.split('.')[0]#+'.'+slide_name.split('.')[1]

def process(opts):
    # global lock
    slide_path, patch_size, patches_output_dir, mask_path, slide_id, max_patches_per_slide = opts
    extract_patches(slide_path, mask_path, patch_size,
                    patches_output_dir, slide_id, max_patches_per_slide)

### Input arguments
####################

parser = argparse.ArgumentParser(description='Generate patches from a given folder of images')
parser.add_argument('--wsi_path', required=True, metavar='WSI_PATH', type=str,
                    help='Path to the input directory of WSI files')
parser.add_argument('--svs_file', required=False, default=None, metavar='SVS_FILE', type=str,
                    help='Path to the svs file')
parser.add_argument('--patch_path', required=True, metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--mask_path', required=True, metavar='MASK_PATH', type=str,
                    help='Path to the  directory of numpy masks')
parser.add_argument('--patch_size', default=224, type=int, help='patch size, default 224')
parser.add_argument('--max_patches_per_slide', default=4000, type=int)
parser.add_argument('--num_process', default=10, type=int,
                    help='number of mutli-process, default 10')
parser.add_argument('--dezoom_factor', default=1.0, type=float,
                    help='dezoom  factor, 1.0 means the images are taken at 20x magnification, 2.0 means the images are taken at 10x magnification')

### MAIN
##########
if __name__ == '__main__':

    args = parser.parse_args()

    slide_list = []
    if args.svs_file is None:
        slide_list = os.listdir(args.wsi_path)
        slide_list = [s for s in slide_list if s.endswith('.svs')]
    else:
        slide_list.append(args.svs_file)
    #slide_list = ['x067.svs']

    opts = [
        (os.path.join(args.wsi_path, s), (args.patch_size, args.patch_size), args.patch_path, args.mask_path,
         get_slide_id(s), args.max_patches_per_slide) for
        (i, s) in enumerate(slide_list)]
    pool = Pool(processes=args.num_process)
    pool.map(process, opts)