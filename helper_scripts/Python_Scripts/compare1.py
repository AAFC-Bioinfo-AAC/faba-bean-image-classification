# Case study to compare the segmentation methodologies in SAM, openCV and Sci-kit libraries for Faba beans image classification pipeline

#__author__="harpreet kaur bargota"
#__email__="harpreet.bargota@agr.gc.ca"
#__Project__="WGRF - Image Classification pipeline for Faba beans"

#References: 
#SegmentAnthing (MetaAI): https://github.com/facebookresearch/segment-anything 
# Reference paper: @article{kirillov2023segany, title={Segment Anything}, author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},journal={arXiv:2304.02643},year={2023}}

#### installation steps- Follow the steps to install all the required packages
#Step1: conda create -n <env-name>
#Step2: conda install opencv
#step3:conda install conda-forge::matplotlib
#Step4:conda install anaconda::seaborn
#step5:pip install pandas
#Step6:conda install numpy
#Step7:conda install anaconda::scikit-image
#pip install git+https://github.com/facebookresearch/segment-anything.git
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth



#Import the libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
import os
import pandas as pd
import seaborn as sns
from skimage.util import crop
from skimage.filters import try_all_threshold
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
#import torch
#from pycocotools import mask as mask_utils
from typing import Any, Dict, List, Optional, Tuple
import sys
sys.path.append("..")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor




#Defining the functions for plotting the SAM masks
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Create a folder to save the images if it doesn't exist
output_folder = 'comparison_study'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)




# Read the image
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.normpath(os.path.join(script_dir, '..', '..', 'Data', 'Faba-Seed-CC_Vf1-1-2.JPG'))

if os.path.exists(img_path):
    image = cv2.imread(img_path)
    if image is None:
        print("Failed to load image with OpenCV (corrupt/unsupported format)")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
else:
    print("Image path does not exist:", img_path)

#Crop the upperpart of the image having the colorcard
image = crop(image, ((2000, 800), (50, 50), (0,0)), copy=False)



#Image Segmenation using SegmentAnything MetaAI
#SAM dependencies
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

#SAM model and checkpoint, device
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) 
sam.to(device=device) 
mask_generator = SamAutomaticMaskGenerator(sam)

#generates masks
masks = mask_generator.generate(image)  
print(len(masks))

#Save SAM maks in output folder

for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"SAM_{i}.png"
        cv2.imwrite(os.path.join(output_folder, filename), mask * 255)

######################################################################################################################

#Convert the image to gray scale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding methods in OpenCV
threshold_methods = [
    (cv2.THRESH_BINARY, 'Binary_Global_Thresholding'),
    (cv2.THRESH_BINARY_INV, 'Binary_Inverted_Global_Thresholding'),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 'Binary_Adaptive_Mean_Thresholding'),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 'Binary_Adaptive_Gaussian_Thresholding')
]   #

for method, method_name in threshold_methods:
    if method in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, binary_image = cv2.threshold(image, 127, 255, method)
    else:
        _, binary_image = cv2.adaptiveThreshold(image, 255, method, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    
    # Plot and save the image
    plt.imshow(binary_image, cmap='gray')
    plt.title(method_name)
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f'{method_name}.png'))
    plt.close()

#####################################################################################################################
# Thresholding methods in scikit-image
threshold_methods = [
    ('otsu', 'Binary_Global_Thresholding_Scikit'),
    ('local', 'Binary_Local_Thresholding')
]

for method, method_name in threshold_methods:
    if method == 'otsu':
        binary_image = image > filters.threshold_otsu(image)
    else:
        binary_image = image > filters.threshold_local(image, block_size=51)
    
    # Plot and save the image
    plt.imshow(binary_image, cmap='gray')
    plt.title(method_name)
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f'{method_name}.png'))
    plt.close()

# All threshold methods in Scikit :
fig, ax = try_all_threshold(image, figsize=(10, 8), verbose=False)
plt.show()
plt.savefig(os.path.join(output_folder, "Scikit_threshold.jpg"))

