### Prerequisites
•	**Programming Knowledge**: Familiarity with Python and Linux

•	**Libraries**: PyTorch, OpenCV, Sci-kit image, Numpy, Pandas, Matplotlib, Circle-fit and SAM2’s official repository.

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
│-- helper_scripts   # Scripts for data analysis and images
```

## Installation

### Using Conda (Recommended)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
2. Create a new Conda environment using environment.yml file 

2a. Navigate to the directory containing environment.yml

   ```bash
   cd /path/to/environment.yml
   ```
2b. Run the following command to ensure that defaults is explicitly set in your Conda configuration.

   ```bash
   conda config --add channels defaults
   ```
2c. Create the Conda Environment

   ```bash
   conda env create -f environment.yml
   ```
2d. Activate the environment:

   ```bash
   conda activate fababean_env
   ```

3. Clone SAM 2 Github Repository, checkpoints and Step1_SAM2.1.py 

3a.	Clone SAM 2 github repository and download the checkpoints by running the commands as:

   ```bash
   git clone https://github.com/facebookresearch/sam2.git && cd sam2
   pip install -e .
   cd checkpoints && \
   ./download_ckpts.sh && \
   cd ..
   ```
3b.	Copy the python script in the sam2 directory

   ```bash
   cp ../Step1_SAM2.1.py .
   ```
3c.	change the directory to the parent directory

   ```bash
   cd ..
   ```

## Usage   

Note: After installation, the parent directory contains two subdirectories, image directory with .JPG images (e.g. faba_images)  and sam2, as well as the python scripts: Step1_SAM2.1.py, Step2_SAM2.1.py and Step3_color.py. The subdirectories and the .ipynb file are all located at the same hierarchical level within the parent directory. The subdirectory sam2 also contains Step1_SAM2.1.py.

4. For running the pipeline, follow the steps 1-3 

> **Note:**  
> In the following steps:
> - `<nameofinputdir>` refers to the directory containing the input images.
> - `<nameofoutputdir>` refers to the directory where the binary masks and metadata file will be saved.
> - `<nameofnewoutputdir>` refers to the directory where the output will include:
>     - A `.csv` file with dimensional and shape features.
>     - Seed count information.
>     - Annotated binary images.
>
> Only the names of the output folders `<nameofoutputdir>` and `<nameofnewoutputdir>` need to be specified in the CLI.  
> 
> For example:  
> - `<nameofoutputdir>` = `output_SAM`  
> - `<nameofnewoutputdir>` = `output_FE`
>
> ⚠️ Do not manually create the output directories. They will be automatically created in the parent directory as part of the code execution.


Step1: Generation of binary masks from images folder

Python script Step1_SAM2.1.py takes the images as input and generates the binary masks (.png) and metadata .csv file for each image using SAM2.1 model in the output directory. Run the following command for generating masks and metadata file

   ```bash
   python sam2/Step1_SAM2.1.py <nameofinputdir> <nameofoutputdir>
   ```
5. Step2: Extraction of dimensional and shape features (.csv file) and seed count (.xlsx file) from binary masks and metdata file from the output of Step1: 

The python script (Step2_SAM2.1.py) uses the binary masks and metadata (from output of Step1) as input and generates the .csv file of dimensional & shape features and binary annotated combined masks (.png) as output in another output folder. Run the following command for generating the output files:

   ```bash
   python Step2_SAM2.1.py <nameofoutputdir> <nameofnewoutputdir>
   ```

6. Step3: Color extraction from images and features extraction files

The python script (Step3_color.py), takes the .csv file of dimensional and shape features and images as inputs to generate output as feature extraction containing the dimensional, shape features, RGB values and color in the .csv file. 

   ```bash
   python Step3_color.py <nameofinputdir> <nameofnewoutputdir>
   ```


###🎯 CLI : Example for running steps (Step1,2 and 3) of pipeline:

•	faba_images: Input directory of images
•	output_SAM: Output dir with masks and metadata file
•	output_FE: Output dir with dimensional & shape features, color RGB values, Color label, seed count (.csv, .xlsx)

(4.,5.,6.) Run the CLI as:

   ```bash
   python sam2/Step1_SAM2.1.py faba_images output_SAM

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
