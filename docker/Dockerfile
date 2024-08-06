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

# Set the env var to disable oneDNN optimization
ENV TF_ENABLE_ONEDNN_OPTS=0

# Copy the platform directory to the Docker image
COPY fr /fr
COPY docker/engine-fr-services.sh /engine-fr-services.sh

WORKDIR /fr

RUN conda create -y --name fr python=3.12

# install cudnn
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2004-9.3.0_1.0-1_amd64.deb
RUN dpkg -i cudnn-local-repo-ubuntu2004-9.3.0_1.0-1_amd64.deb
RUN cp /var/cudnn-local-repo-ubuntu2004-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cudnn

# Activate the environment and install Python packages
RUN /bin/bash -c \
    "source activate fr && \
    bash /fr/env/install.sh"

# Make sure the "deep" environment is activated in every shell
RUN echo "source /opt/conda/bin/activate fr" >> ~/.bashrc

# Expose FastAPI application port

# Set the default command
CMD ["/engine-fr-services.sh", "/bin/bash"]