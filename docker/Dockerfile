# Use the official Ubuntu 20.04 image as the base image
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    lsb-release \
    gnupg2 \
    software-properties-common \
    python3.8 \
    python3.8-dev \
    python3-pip \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the latest Miniconda (flexible with Python version)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Set PATH to include conda
ENV PATH=/opt/conda/bin:$PATH

# Copy the platform directory to the Docker image
COPY fr /fr
COPY docker/engine-fr-services.sh /engine-fr-services.sh

WORKDIR /fr

RUN conda create -y --name fr python=3.8

# Install NVIDIA's version of TensorFlow
RUN conda install -y -c conda-forge cudatoolkit=12.2 cudnn=8.9 && \
    pip install tensorflow==2.13.0

# Activate the environment and install Python packages
RUN /bin/bash -c \
    "source activate fr && \
    bash /fr/env/install.sh"

# Make sure the "deep" environment is activated in every shell
RUN echo "source /opt/conda/bin/activate fr" >> ~/.bashrc

# Expose FastAPI application port

# Set the default command
CMD ["/engine-fr-services.sh", "/bin/bash"]