#hkb
#__author__="harpreet kaur bargota"
#__email__="harpreet.bargota@agr.gc.ca"
#__Project__="Faba bean Feature extraction pipeline (Step3)"
#References:
#https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2lab
#https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors
#https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

"""This Step3 processes images from folder based on bounding box (bbox-0,1,2,3) coordinates stored in a .csv file of dimensional & shape features from output directory of step2.
It converts the normalized RGB value to the closest named color from the CSS4 color set using the CIE Lab color space for better perceptual accuracy.
 and counts the RGB color name. The maximum count of color name (excluding the blue background) is the dominant color and the dominant RGB value. Finally
  new columns for dominant RGB value and dominant color of the bean is cretaed in the in the dimensional and shape features dataframe, and also the average RGB value is calculated and a new column for avg RGB value is created in the date frame, which is 
  saved to a .csv file in the output directory."""


import os
import cv2
import numpy as np
import pandas as pd
import argparse
from skimage.color import rgb2lab
from collections import Counter
from scipy.spatial.distance import cdist
from matplotlib.colors import CSS4_COLORS

# Function to map an RGB color to a CSS4 color name
def rgb_to_css4_color_name(rgb):

    """ Converts the normalized RGB value to the closest named color from the CSS4 color 
        set using the CIE Lab color space for better perceptual accuracy. 
            
        Arguments: RGB value """    

    rgb_normalized = np.array(rgb) / 255  # Normalize to 0-1 scale
    lab_color = rgb2lab(np.array([[rgb_normalized]]))[0][0]
    
    css_colors_lab = np.array([
        rgb2lab(np.array([[np.array([
            int(c[1:3], 16), int(c[3:5], 16), int(c[5:], 16)]) / 255.0]]))[0][0]
        for c in CSS4_COLORS.values()
    ])
    
    colors_distances = cdist([lab_color], css_colors_lab)
    closest_color_index = np.argmin(colors_distances)
    return list(CSS4_COLORS.keys())[closest_color_index]

# Function to find the most frequent RGB tuple in an ROI, excluding blue background
def get_dominant_color_excluding_blue(image, x, y, w, h):
    """Extracts the most dominant color i.e. the maximum count of color name from bounding box coordinates of bean boxes,
         while excluding blue background color (specifically "cornflowerblue" and "dodgerblue").
         
         Argumnrts: image and bounding box coordinates x,y,w,h"""

    roi = image[int(y):int(y + h), int(x):int(x + w)]  # Crop region
    roi_reshaped = roi.reshape(-1, 3)  # Flatten
    color_counts = Counter(map(tuple, roi_reshaped)).most_common()

    for color, _ in color_counts:
        color_name = rgb_to_css4_color_name(color)
        if color_name.lower() not in ["cornflowerblue", "dodgerblue"]:
            return color  

    return color_counts[0][0]  # If all are blue, return most common

# Function to compute average RGB values inside a mask (excluding background)
def get_average_rgb(image, mask, x, y, w, h):
    """Extracts the average R,G,B values from  the image, mask and bounding box coordinates
         
         Arguments: image, mask and bounding box coordinates x,y,w,h"""

    roi_image = image[int(y):int(y + h), int(x):int(x + w)]
    roi_mask = mask[int(y):int(y + h), int(x):int(x + w)]

    object_pixels = roi_image[roi_mask > 0]  # Select object pixels using mask

    if len(object_pixels) == 0:
        return (0, 0, 0)  # Return black if no object pixels found

    avg_r, avg_g, avg_b = np.mean(object_pixels, axis=0)
    return (int(avg_r), int(avg_g), int(avg_b))

# Main processing function
def process_color(image_folder, output_folder):

    """ Processes images from folder based on bounding box (bbox) coordinates stored in a .csv file of dimensional & shape features.
          It extracts the dominant RGB value and the color of the specified regions in each image while excluding blue background shades and saves the results in new column
          in the dataframe which is saved to a .csv file in the output directory.
          
          Arguments: Input directory of images, output directory of step2 which contains the .csv file of dimensional and shape features

          Raises:
            TypeError: If the conditions are not met.

          """

    csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No .csv file found in folder: {output_folder}")

    csv_path = os.path.join(output_folder, csv_files[0])
    df_image = pd.read_csv(csv_path)

    color_names = []
    RGB_values = []
    avg_R_values, avg_G_values, avg_B_values = [], [], []
    avg_color_names = []

    for index, row in df_image.iterrows():
        x = row["bbox-1"]
        y = row["bbox-0"]
        w = row["bbox-3"] - row["bbox-1"]
        h = row["bbox-2"] - row["bbox-0"]

        class_name = row['Class']
        image_path = os.path.join(image_folder, f"{class_name}.JPG")
        mask_path = os.path.join(output_folder, f"{class_name}_mask.png")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        dominant_color = get_dominant_color_excluding_blue(image, x, y, w, h)
        RGB_values.append(dominant_color)
        color_names.append(rgb_to_css4_color_name(dominant_color))

        avg_r, avg_g, avg_b = get_average_rgb(image, mask, x, y, w, h)
        avg_R_values.append(avg_r)
        avg_G_values.append(avg_g)
        avg_B_values.append(avg_b)

        avg_color_name = rgb_to_css4_color_name((avg_r, avg_g, avg_b))
        avg_color_names.append(avg_color_name)

    df_image["RGB value of Seed"] = RGB_values
    df_image["color_seeds"] = color_names
    df_image["Avg_R"] = avg_R_values
    df_image["Avg_G"] = avg_G_values
    df_image["Avg_B"] = avg_B_values
    df_image["color_avg_seeds"] = avg_color_names  # Color name based on average RGB

    output_filename = "FE_Color.csv"
    output_path = os.path.join(output_folder, output_filename)
    df_image.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract color features from images and masks")
    parser.add_argument("image_folder", help="Path to the images")
    parser.add_argument("output_folder", help="Path to the output folder")

    args = parser.parse_args()
    process_color(args.image_folder, args.output_folder)
