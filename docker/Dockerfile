# Use the official Ubuntu 20.04 image as the base image
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

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

# Create a conda environment named "fr" with Python 3.8
RUN conda init

# Copy the platform directory to the Docker image
COPY docker/engine-fr-services.sh /engine-fr-services.sh
COPY . /fr

WORKDIR /fr

# Activate the "deep" environment and install Python packages
RUN /bin/bash -c \
    "cd /fr && \
    bash env/env.sh && \
    bash env/activate.sh && \
    bash install.sh"

# Make sure the "deep" environment is activated in every shell
RUN echo "source /opt/conda/bin/activate fr" >> ~/.bashrc

# Expose FastAPI application port

# Set the default command
CMD ["bash"]