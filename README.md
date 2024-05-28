[]([url](url))# WGRF Faba beans Image Classification Pipeline

- Repository to store code and files related to image work

**Introduction:**
The current work involves constructing of image classification pipeline to use faba bean images as the input file, to extract the features from them and make accurate class predictions based on deep learning models. This work involves the use of different machine learning and deep learning models such as SegmentAnything (SAM), OpenCV, Sci-kit, supervision and CNN classification. 
The novelty of this work lies in the utilization of SegmentAnything for image segmentation. While researchers have traditionally relied on OpenCV and scikit-image libraries for segmentation tasks, this study leverages SegmentAnything to produce high-quality masks. 

**SegmentAnything:**
The Segment Anything Model (SAM) excels at generating high-quality object masks from input prompts like points or boxes and can produce masks for all objects within an image. Trained on a vast dataset of 11 million images and 1.1 billion masks, SAM demonstrates robust zero-shot performance across various segmentation tasks.
This innovative approach ensures more precise and detailed segmentation, marking a significant advancement over conventional methods.

![Feature_Extraction Flowchart](https://001gc-my.sharepoint.com/:i:/r/personal/harpreet_bargota_agr_gc_ca/Documents/Pictures/Picture1.jpg?csf=1&web=1&e=a115Gl?raw=true "Title")

**Current Status:**
Currently, 254 images of fava beans (Vicia faba) provided by Dr. Nicholas Larkan have been used/tested for development of an image analysis pipeline for extracting 14 Dimensional features (Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter, Extent, Solidity, Eccentricity, Aspect ratio, Compactness, Roundness) and 4 shape factors (Shapefactor1-4) from images. For using SegmentAnything for this pipeline, initially some research work was done for comparing the segmenatation stratesies of SAM and other conventional libraries as OpenCV and Sci-kit. Furthermore, to test the effeciacy of different segmentation approaches in SAM, some research tests were done to conclude the utilization of automatic mask generation method of SAM to be used for segmentation purposes. 

**Installation:**
This work is being done at Lethbridge Superdome and AWS Phenomics sandbox.
