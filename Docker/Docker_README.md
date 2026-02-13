# Docker Usage: Faba Bean Feature Extraction Pipeline

## 1. Overview
This containerized pipeline automates the segmentation and feature extraction of Faba bean images. It encapsulates the complete Python environment, including **Segment Anything Model 2.1 (SAM 2)**, PyTorch, and OpenCV, configured to run on standard **CPU hardware**.

**What this container does:**
1.  **Segmentation:** Uses SAM 2.1 (Small Model) to generate binary masks for every bean in an image.
2.  **Feature Extraction:** Calculates morphological traits (Area, Perimeter, Shape Factors, Circularity).
3.  **Color & Weight Analysis:** Calibrates color using a reference card and predicts Thousand Grain Weight (TGW).

---

## 2. System Requirements
While this image runs on a CPU (no GPU required), AI image segmentation is memory-intensive.

* **OS:** Windows, macOS, or Linux (with Docker Desktop or Docker Engine installed).
* **CPU:** Any modern multi-core processor (x86_64).
* **RAM (Crucial):**
    * **Minimum:** 16GB System RAM.
    * **Recommended:** 32GB System RAM.
* **Disk Space:**
    * ~7GB for the Docker Image.
    * Additional space for your input/output images.

> Note on Image Resolution and Accuracy
This pipeline is designed to process high-resolution input images (standard 4000x6000px). To ensure stability on standard hardware, the Docker container automatically downscales these images by 50% during processing.

Because of this downscaling, raw pixel measurements will differ from the full, non-containerized version of the pipeline. While ratio-based metrics (mm, mm2) remain comparable, exact results may not be 1:1 identical to the main development build. This Docker image is intended as a portable proof of concept. **Please do not pre-resize your images before running the container.**

---

## 3. Quick Start (Running the Pipeline)

### Execution Steps

1.  **Create your folders:**
    Ensure you have created the `output_dir` folder yourself *before* running the command. If Docker creates it for you, it may have incorrect permissions.
    ```bash
    mkdir -p input_dir output_dir
    ```

2.  **Run the command:**
    
    **Linux / macOS:**
    ```bash
    docker run --rm \
        --shm-size=2g \
        --user "$(id -u):$(id -g)" \
        -v "$(pwd)/input_dir":/data/input \
        -v "$(pwd)/output_dir":/data/output \
        -e HOME=/tmp \
        -e PYTHONDONTWRITEBYTECODE=1 \
        faba-pipeline:v1 /data/input /data/output
    ```

    **Windows (PowerShell):**
    ```PowerShell
    docker run --rm `
        --shm-size=2g `
        -v "${PWD}\input_dir:/data/input" `
        -v "${PWD}\output_dir:/data/output" `
        faba-pipeline:v1 /data/input /data/output
    ```

**Understanding the command flags:**
- `--rm`: Automatically deletes the container after the job finishes (to save disk space).
- `--shm-size=2g`: Increases shared memory to prevent PyTorch crashes.
- `--user ...`: (Linux only) Runs the container with your user ID so output files are not locked by 'root'.
- `-v`: Volume Mapping. This links a folder on your computer to a directory inside the container.
    - Format: `-v host_folder:container_folder`
- `/data/input /data/output`: Arguments passed to the script telling it where to look inside the container.

---

## 4. Expected Outputs

After the run completes, check your local `output_dir`. You will find:

| Folder / File | Description |
|---|---|
| step1_masks/ | Contains binary PNG masks for every detected seed |
| step2_features/ | Intermediate CSVs containing raw pixel and metric measurements |
| FE_Color.csv | **The final results**. Contains all shape features, color classes, and TGW predictions |

---

## 5. Building the Image

If you want to modify the code or build the image from source instead of pulling it from a registry:

1. Navigate to the project directory containing the Dockerfile.
2. Run the build command:

```bash
docker build -t faba-pipeline:v1 .

---

## 6. Troubleshooting

Error: "Killed" or "Exit Code 137"

- Cause: The container ran out of RAM
    - Solution: Increase the memory allocated to Docker

Error: "Path not found"

- Cause: The file path in the `-v` flag is incorrect.
    - Solution: Use absolute paths (eg. `/home/user/data` instead of `./data`)

