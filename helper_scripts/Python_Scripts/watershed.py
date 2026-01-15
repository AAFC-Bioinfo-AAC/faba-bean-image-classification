## Comparison of SAM, opencv for studying the segmenatation methods
# Watershed segmentation method of opencv was used to test the segmentation effeciency on faba bean image.
#Create an Input folder, (e.g. 'SAM_compare'), which contains the image ('Faba-Seed-CC_Vf1-1-2.JPG') and the results ('Watershed_opencv') have been saved in the same input folder.
#Run the code to get the results


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


# Load the image
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.normpath(os.path.join(script_dir, '..', '..', 'Data', 'Faba-Seed-CC_Vf1-1-2.JPG'))

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Crop the upperpart of the image having the colorcard
image = crop(image, ((2000, 800), (50, 50), (0,0)), copy=False)

original_image = image.copy()  # Keep a copy of the original image for later use
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#opencv-watershed algorithm (https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html)
# Apply thresholding
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]  # Mark boundaries in red

# Draw contours
contours, _ = cv2.findContours(markers.copy().astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 1:  # Filter out small contours if needed
        cv2.drawContours(original_image, contours, i, (255, 255, 255), 2)  # Draw each contour in white

# Plot the results
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(232), plt.imshow(gray, cmap='gray'), plt.title('Gray Image')
plt.subplot(233), plt.imshow(binary, cmap='gray'), plt.title('Binary Image')
plt.subplot(234), plt.imshow(sure_bg, cmap='gray'), plt.title('Sure Background')
plt.subplot(235), plt.imshow(sure_fg, cmap='gray'), plt.title('Sure Foreground')
plt.subplot(236), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Contours')
plt.tight_layout()
plt.show()

# Save the plot to the same directory as the script
output_path = os.path.join(script_dir, 'Watershed_opencv.png')  # add extension
plt.savefig(output_path)
print(f"Plot saved to {output_path}")


