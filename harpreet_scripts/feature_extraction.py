#hkb
#__author__="harpreet kaur bargota"
#__email__="harpreet.bargota@agr.gc.ca"
#__Project__="WGRF - Image Classification pipeline for Faba beans"

#References: 
#SegmentAnthing (MetaAI): https://github.com/facebookresearch/segment-anything 
# Reference paper: @article{kirillov2023segany, title={Segment Anything}, author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},journal={arXiv:2304.02643},year={2023}}

# Supervision library for image annotation: Reference: https://supervision.roboflow.com/annotators/#polygonannotator

#Feature extraction: 
#scikit-image library for image processing: Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart, Tony Yu and the scikit-image contributors. scikit-image: Image processing in Python. PeerJ 2:e453 (2014) https://doi.org/10.7717/peerj.453
#https://scikit-image.org/docs/stable/api/skimage.measure.html



###Image classification pipeline using amg SAM, feature extraction and classification (to be done)

    
    ## Import the required libraries

import argparse
import os as os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import skimage
import supervision as sv

from skimage.util import crop
from skimage import measure
from skimage.measure import label, regionprops, regionprops_table

from typing import Any, Dict, List, Optional, Tuple
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")



#Dependencies for SAM

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"    


# defining the function for processing of images


def process_images(input_folder, output_folder):
    # Check if the input folder exists
	if not os.path.exists(input_folder):
		print(f"Error: Input folder '{input_folder}' does not exist.")
		return

    # Create the output folder if it doesn't exist
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
		print(f"Output folder '{output_folder}' created.")

	
	
	#Iteration loop for processing images
	for root, dirs, files in os.walk(input_folder):
		df_total = []
		
		for idx, file in enumerate(files):

#Step1:        # Read an image
			image_path=os.path.join(root,file)
			image=cv2.imread(image_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
			image = crop(image, ((2200, 500), (0, 0), (0,0)), copy=False)
			

#Step2: 	#Apply SegmentAnything model 
 
			sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) 
			sam.to(device=device) 
			mask_generator = SamAutomaticMaskGenerator(sam, box_nms_thresh = 0.1, output_mode = "binary_mask") 
			masks_SAM = mask_generator.generate(image)  
			print ("The total length of all masks is: ", len(masks_SAM))
			
			
            # Metadata
			
			df1=[]
			header1=['id','area','bbox_x0','bbox_y0','bbox_w','bbox_h','point_input_x','point_input_y','predicted_iou','stability_score','crop_box_x0','crop_box_y0','crop_box_w','crop_box_h']
			metadata = []
			for i, mask_data in enumerate(masks_SAM):
				mask = mask_data["segmentation"]
				mask_metadata = [
					str(i),
					str(mask_data["area"]),
					*[str(x) for x in mask_data["bbox"]],
					*[str(x) for x in mask_data["point_coords"][0]],
					str(mask_data["predicted_iou"]),
					str(mask_data["stability_score"]),
					*[str(x) for x in mask_data["crop_box"]],
					]    #create a list of metadata with header1
				df=pd.DataFrame(mask_metadata, index=header1)
				df1.append(df)
			df2=pd.concat(df1,axis=1)
			df_metadata=df2.T.apply(pd.to_numeric).set_index('id')
			
 #Step3:           
    # Filtering the masks of beans:

			# dataframe for coin required for standardization of area, length and width
			df_metadata_coin = df_metadata[(df_metadata['bbox_x0'] >= 3300) & (df_metadata['area'] >= 200000)]
			# Find the mask of coin
			Mask_index= df_metadata_coin.index.tolist()
			masks_coin=[masks_SAM[x] for x in Mask_index]
			print ('index of coin mask is ', Mask_index)
			
			#Eliminating the masks of colorcard, label, scale, coin, and duplicate masks for seeds
			conditions = [(df_metadata['bbox_x0'] <= 1900) & (df_metadata['bbox_y0'] >= 2500), #label
    		(df_metadata['bbox_x0'] <= 4000) & (df_metadata['bbox_y0'] >= 3000), #Scale
    		(df_metadata['bbox_x0'] >= 3300) & (df_metadata['bbox_y0'] >= 2400), #coin
   			(df_metadata['bbox_h'] >= 650), #duplicate mask for seed
    		(df_metadata['bbox_w'] >= 650) #duplicate mask for seed
			]
			
			for condition in conditions:
				df_metadata = df_metadata.drop(df_metadata[condition].index)
			print (df_metadata)

			



	    	# Saving the metadata for the masks of beans to the output folder
			image_filename = os.path.basename(image_path)
			output_filename = f"Metadata_{image_filename}.xlsx"
			output_path= os.path.join(output_folder, output_filename)
			df_metadata.to_excel(output_path)

						
		# Save fava bean masks in output_folder
			Masks_list=df_metadata.index.tolist()
			masks_SAM=[masks_SAM[x] for x in Masks_list]
			for i, mask_data in enumerate(masks_SAM):
				mask = mask_data["segmentation"]
				image_filename = os.path.basename(image_path)
				output_filename = f"{image_filename}_mask_{idx}_{i}.png"
				output_path= os.path.join(output_folder, output_filename)
				cv2.imwrite(output_path, mask*255)
			
			print("After filtering of the masks, now the length of all masks required for feature extraction is: ", len(masks_SAM))
			
#Step4:            ##### IMAGE ANNOTATION --  Supervision detections
			
			# Reference: https://supervision.roboflow.com/annotators/#polygonannotator
			detections = sv.Detections.from_sam(sam_result=masks_SAM)
			polygon_annotator = sv.PolygonAnnotator(color=sv.ColorPalette.ROBOFLOW, thickness=36, color_lookup=sv.ColorLookup.INDEX)
			annotated_frame = polygon_annotator.annotate(scene=image.copy(),	detections=detections)
			image_filename = os.path.basename(image_path)
			output_filename = f"Annotated_image_{image_filename}.png"
			output_path= os.path.join(output_folder, output_filename)
			cv2.imwrite(output_path, annotated_frame)


			#Standardize 
			df_coin=[]
			for i, mask_data in enumerate(masks_coin):
				mask=mask_data["segmentation"]
				label_image = measure.label(mask)
				props = regionprops_table(label_image,properties=('area','perimeter','axis_major_length', 'axis_minor_length')) 
				coin_std = pd.DataFrame(props)

				Length_coin_mm=23.88
				width_coin_mm=23.88

				Area_standard_coin_pixels= coin_std.iloc[0]['area'] # Area of coin in pixels
				Area_Standard_coin_mm2=3.14*(Length_coin_mm/2)*(Length_coin_mm/2) # Area of coin in mm2
				Calibration_factor_area=(Area_Standard_coin_mm2/Area_standard_coin_pixels) # Calibration factor for area

				axis_major_length_pixels= coin_std.iloc[0]['axis_major_length'] # Length of coin in pixels
				Calibration_factor_length=(Length_coin_mm/axis_major_length_pixels) # Calibration factor for length

				axis_minor_length_pixels= coin_std.iloc[0]['axis_minor_length'] # Width of coin in pixels
				Calibration_factor_width=(width_coin_mm/axis_minor_length_pixels) # Calibration factor for width

				perimeter_mm =(2*3.14*Length_coin_mm)/2
				perimeter_pixels= coin_std.iloc[0]['perimeter'] # perimeter of coin in pixels
				Calibration_factor_perimeter=(perimeter_mm/perimeter_pixels) # Calibration factor for perimeter
#Step5:			
# FEATURE EXTRACTION
			df_list=[]
			for i, mask_data in enumerate(masks_SAM):
				mask = mask_data["segmentation"]
			
				#Label the mask
				#Reference: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_label.html
				label_image = measure.label(mask)

				#analyse masks
				#Reference: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
				props = regionprops_table(label_image,properties=('centroid_local', 'orientation', 'area','eccentricity', 'equivalent_diameter_area','perimeter','solidity', 'area_convex', 'extent','axis_major_length', 'axis_minor_length'),) #properties = measure.regionprops(label_image, intensity_image=image)
				
				# convert to dataframe
				df_FE = pd.DataFrame(props)
				
				#For calculating the other parameters, mathematical operations have been used as advised in paper(Multiclass classification of dry beans using computer vision and machine learning techniques)
				df_FE["Area_mm2"]=df_FE["area"] * Calibration_factor_area
				df_FE["Length_mm"]=df_FE["axis_major_length"] * Calibration_factor_length
				df_FE["Width_mm"]=df_FE["axis_minor_length"] * Calibration_factor_width
				df_FE["perimeter_mm2"]=df_FE["perimeter"] * Calibration_factor_perimeter
				
				df_FE["Aspect_Ratio"] = df_FE["axis_major_length"]/df_FE['axis_minor_length']
				df_FE["Roundness"]= (4*3.14*(df_FE["area"]))/((df_FE["perimeter"])**2)
				df_FE["Compactness"] = df_FE["equivalent_diameter_area"]/df_FE["axis_major_length"]
				
				#Shape features
				df_FE["Shapefactor1"] = df_FE["axis_major_length"]/df_FE["area"]
				df_FE["Shapefactor2"] = df_FE["axis_minor_length"]/df_FE["area"]
				df_FE["Shapefactor3"] = (df_FE["area"])/(((df_FE["axis_major_length"])/2)*((df_FE["axis_major_length"])/2)*3.14)
				df_FE["Shapefactor4"] = (df_FE["area"])/(((df_FE["axis_major_length"])/2)*((df_FE["axis_minor_length"])/2)*3.14)

				# The name of class has been extracted from the image file name 
				class_in_image=(image_path).split('.JPG')[0]
				df_FE["class"]= (image_path).split('.JPG')[0]
				df_list.append(df_FE)
			
			df_FE2 = pd.concat(df_list)
			df_total.append(df_FE2)
		df_image=pd.concat(df_total)
		print(df_image)
 


#Save the final file of feature extraction from all images into output folder
		output_filename = f"Fava_bean_Features_extraction.xlsx"
		output_path= os.path.join(output_folder, output_filename)
		df_image.to_excel(output_path)
		output_filename = f"Fava_bean_Features_extraction.csv"
		output_path= os.path.join(output_folder, output_filename)
		df_image.to_csv(output_path)
		print ("Feature extraction from fava bean images is completed..!")





if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Process files from input folder to output folder.")

    # Add arguments for input and output folders
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("output_folder", help="Path to the output folder")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the process_folders function with provided input and output folders
    process_images(args.input_folder, args.output_folder)

#for x in args.input_folder:
#	image = cv2.imread(r'x')   # read an image using openCV





