# Base: Lightweight Python 3.10 on Debian
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies
# ffmpeg/libsm6/libxext6 for OpenCV; git/wget for fetching assets
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Workspace Setup
WORKDIR /app

# 3. Install PyTorch (CPU)
# Installed explicitly first to leverage Docker layer caching for this heavy dependency
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install Python Requirements
COPY frozen_requirements.txt /app/frozen_requirements.txt
RUN pip install --no-cache-dir -r /app/frozen_requirements.txt

# 5. Install SAM 2 (Pinned Commit)
# Clones official repo, checks out specific commit for stability, and installs in editable mode
RUN git clone https://github.com/facebookresearch/sam2.git && \
    cd sam2 && \
    git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4 && \
    pip install -e .

# 6. Download Model Checkpoint (Small)
# Places weights exactly where Step 1 script expects them
RUN mkdir -p /app/sam2/checkpoints && \
    wget -q -P /app/sam2/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# 7. Copy Pipeline Scripts
COPY scripts/ /app/scripts/
COPY run_pipeline.sh /app/run_pipeline.sh
RUN chmod +x /app/run_pipeline.sh

# 8. Execution Entrypoint
ENTRYPOINT ["/app/run_pipeline.sh"]