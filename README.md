# Faba bean feature extraction pipeline from WGRF-faba bean images

## Overview
This work provides a workflow for running faba bean feature extraction pipeline to extract the dimensional, shape and color of faba bean seeds in the .csv file from the faba bean images. It presents a methodology for seed image segmentation and feature extraction using advanced deep learning and image processing techniques. The [Segment Anything Model 2.1](https://github.com/facebookresearch/sam2/blob/main/README.md) (SAM2.1) has been used for precise segmentation, while [OpenCV](https://docs.opencv.org/4.x/d7/dbd/group__imgproc.html), [Scikit-Image](https://scikit-image.org/), and [Matplotlib-colors](https://matplotlib.org/stable/gallery/color/named_colors.html) are employed to analyze the dimensional, spatial, shape, and color properties of segmented seeds. The pipeline also gives the seed count in an image and annotated binary images. The pipeline has been specifically developed based on the spatial coordinates of faba bean seeds, colorcard, label, ruler and coin.

### Faba bean Images
The images of faba beans were captured according to the Standard Operating Protocol (Figure 1).

<img src="https://gccode.ssc-spc.gc.ca/lethbridge-carsu/wgrf-cloud-phenomics/faba-bean-image-classification/-/raw/main/harpreet_scripts/Images/Faba-Seed-CC_Vf1-1-2.JPG" alt="Figure 1" width="200">

Figure 1. Example of Faba bean images Vf1-1-2 (image shape=6000, 4000, 3) with faba bean seeds, colorcard, coin, label and ruler     


### Segmentanything 2.1 (MetaAI) Model used for image segmentation

[Segment Anything Model 2](https://ai.meta.com/sam2/) (SAM 2.1) is an advanced segmentation model designed to work seamlessly with both images and videos, treating a single image as a one-frame video. This work introduces a new task, model, and dataset aimed at improving segmentation performance. SAM 2 trained on SA-V dataset provides strong performance across a wide range of tasks. In image segmentation, SAM2 model is reported to be more accurate and 6 times faster than the Segment Anything Model (SAM). 

## 💡 Uniqueness/Novelty

The novelty of this work lies in the utilization of SegmentAnything 2.1 for image segmentation. While researchers have traditionally relied on OpenCV and scikit-image libraries for segmentation tasks, this study leverages SegmentAnything 2.1 to produce 

## 🔥 A Quick Overview

<img src="https://gccode.ssc-spc.gc.ca/lethbridge-carsu/wgrf-cloud-phenomics/faba-bean-image-classification/-/raw/main/harpreet_scripts/Images/SAM2.1_Flowchart.png" alt="Figure 2" width="800">

Figure 2: Flowchart for Faba bean feature extraction pipeline

## 📝 Details of Steps **(Figure 2)**:

1. **Step1:** Image/Images are used as input and SAM2.1 model generates the binary masks (.png) and metadata file (.csv) for each image in the Output dir SAM

2. **Step2:** The Output dir SAM (from Step2) is used as input for this step and data  analysis, feature extraction using sci-kit image library and feature engineering gives the .csv file with dimensional and shape features in another output dir FE

3. **Step3:** Both the output dir FE (from Step2) and the images (used as input in Step1) will be used as input for this step and the color labels and RGB values will be extracted using colormath library to give .csv file in the same Final output dir FE (from Step2).

## 📚 Final Output Files

After running the faba bean feature extraction pipeline, there will be 2 output directories-
1.	**Output dir SAM** will contain subfolders (Faba-Seed-CC_Vf_N-N_N) with masks (N.png) and metadata file (metadata.csv) for each image. 
2.	**Output dir FE** will contain :
a.	The .csv file of dimensional and shape features (Fava_bean_Features_extraction.csv)
b.	The .csv file of dimensional, shape, RGB values and Color names (FE_Color.csv)
c.	Seed Count (.xlsx) (Seed Count.xlsx)
d.	Annotated Binary image (.png) with contours around beans (Faba-Seed-CC_Vf_N-N_N_combined_mask.png) 

The features that have been extracted through this pipeline are:
1.	**Dimensional features (19)**: Area_mm2_SAM,Length_mm_SAM, Width_mm_SAM, perimeter_mm_SAM, centroid-0, centroid-1,  bbox-0, bbox-1, bbox-2, bbox-3, Area_pix_SAM, Eccentricity, equivalent_diameter_area, perimeter, solidity, area_convex, extent, Axis Major Length(pix)_SAM, Axis Minor Length(pix)_SAM, Aspect_Ratio, Roundness, Compactness, Circularity_SAM
2.	**Shape features (4)**: Shape, Shapefactor1, Shapefactor2, Shapefactor3, Shapefactor4
3.	**Color (2)**: RGB value, color_seeds
4.	**Seed count**: Number of seeds in image

### Prerequisites
•	**Programming Knowledge**: Familiarity with Python and Linux

•	**Libraries**: PyTorch, OpenCV, Sci-kit image, Numpy, Pandas, Matplotlib, Colormath, and SAM2’s official repository.

•	**Hardware**: A GPU with CUDA support is recommended, (but not necessary) for efficient model inference.

•	**Dataset**: Faba bean images captured using SOP 


### Project Structure

```
project-folder/
│-- README.md          # Project documentation
│-- environment.yml    # Conda environment file
│-- Step1_SAM2.1.py    # Script for generating masks and metadata file using SAM2.1 model on images       
│-- Step2_SAM2.1.py    # Script for extracting dimensional and shape features of beans and seed count
│-- Step3_color.py     # Script for extracting color name and RGB value from images
│-- harpreet_scripts   # Scripts for data analysis and images
```

## Installation

### Using Conda (Recommended)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
2. Create a new Conda environment using environment.yml file 
a. Navigate to the directory containing environment.yml

   ```bash
   cd /path/to/environment.yml
   ```
b. Run the following command to ensure that defaults is explicitly set in your Conda configuration.

   ```bash
   conda config --add channels defaults
   ```
c. Create the Conda Environment

   ```bash
   conda env create -f environment.yml
   ```
d. Activate the environment:

   ```bash
   conda activate fababean_env
   ```

3. Clone SAM 2 Github Repository, checkpoints and Step1_SAM2.1.py 

a.	Clone SAM 2 github repository and download the checkpoints by running the commands as:

   ```bash
   git clone https://github.com/facebookresearch/sam2.git && cd sam2
   pip install -e .
   cd checkpoints && \
   ./download_ckpts.sh && \
   cd ..
   ```
b.	Copy the python script in the sam2 directory

   ```bash
   cp ../Step1_SAM2.1.py .
   ```
c.	change the directory to the parent directory

   ```bash
   cd ..
   ```

## Usage   

4. For running the pipeline, follow the steps 1-3 

Step1: Generation of binary masks from images folder

Python script Step1_SAM2.1.py takes the images as input and generates the binary masks (.png) and metadata .csv file for each image using SAM2.1 model in the output directory. Run the following command for generating masks and metadata file

   ```bash
   python sam2/Step1_SAM2.1.py input_dir <nameofinputdir> output_dir <nameofoutputdir>
   ```
5. Step2: Extraction of dimensional and shape features (.csv file) and seed count (.xlsx file) from binary masks and metdata file from the output of Step1: 

The python script (Step2_SAM2.1.py) uses the binary masks and metadata (from output of Step1) as input and generates the .csv file of dimensional & shape features and binary annotated combined masks (.png) as output in another output folder. Run the following command for generating the output files:

   ```bash
   python Step2_SAM2.1.py <nameofoutputdir> <nameofnewoutputdir>
   ```

Note: <nameofoutputdir> is the directory with binary masks and metadata file while <nameofnewoutputdir> should be the name of new output directory which will contain the .csv file of dimensional & shape features, seed count and annotated binary images

6. Step3: Color extraction from images and features extraction files

The python script (Step3_color.py), takes the .csv file of dimensional and shape features and images as inputs to generate output as feature extraction containing the dimensional, shape features, RGB values and color in the .csv file. 

   ```bash
   python Step3_color.py < image_or_folder> <nameofnewoutputdir>
   ```


###🎯 CLI : Example for running steps (Step1,2 and 3) of pipeline:

•	faba_images: Input directory of images
•	output_SAM: Output dir with masks and metadata file
•	output_FE: Output dir with dimensional & shape features, color RGB values, Color label, seed count (.csv, .xlsx)

(4.,5.,6.) Run the CLI as:

   ```bash
   python sam2/Step1_SAM2.1.py input_dir faba_images output_dir output_SAM

   python Step2_SAM2.1.py output_SAM output_FE

   python Step3_color.py faba_images output_FE

   ```



## Dependencies

The project requires the following dependencies:

- Segmentanything 2.1 
- Python (>=3.8)
- OpenCV (opencv-python)
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- git
- pip
- supervision
- openpyxl
- torch
- torchvision
- torchaudio --index-url https://download.pytorch.org/whl/cu118

These dependencies are included in `environment.yml`.


