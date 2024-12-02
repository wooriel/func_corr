# 도커파일이 있는 디렉토리에서 아래 커멘드를 입력하면 됩니다.
# docker build -t starlab:1.0 .
# 아직 run은 안해봄 (mount필요)
# docker run --gpus all -it --name starlab:1.0 starlab_image

# Use an official Cuda image
FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# To supress prompts during downloading software-properties-common
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install sudo
RUN apt-get update && apt install -y sudo

# Update and install Python
RUN apt-get install -y python3.10 python3-pip && \
    apt-get clean

# Set Python3 as the default Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install git
RUN apt-get install -y git

# Install torch
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install pyyaml
RUN pip install pyyaml

# Add a user named staruser
RUN adduser --disabled-password --gecos "" staruser
# Set password as star6 - for later use in terminal
RUN echo "staruser:star6" | chpasswd
RUN usermod -aG sudo staruser
RUN echo "staruser ALL=(ALL:ALL) ALL" >> /etc/sudoers
RUN echo "staruser ALL=(ALL:ALL) ALL" >> /etc/sudoers.d/staruser && \
    visudo -cf /etc/sudoers.d/staruser && \
    chmod 0440 /etc/sudoers.d/staruser

# Passwordless sudo for staruser
RUN echo "staruser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Change user
USER staruser

# Set the working directory in the container
WORKDIR /home/staruser
