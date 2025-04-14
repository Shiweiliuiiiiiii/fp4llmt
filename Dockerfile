FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install deadsnakes and Python 3.11 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    screen \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    ninja-build \
    libnuma-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

#-------------VS code setup-------------------------------------
# Install OpenSSH server for SSH debugging
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE="in users profile"

RUN echo "export VISIBLE=now" >> /etc/profile

RUN echo "ClientAliveInterval 5" >> /etc/ssh/sshd_config

# Expose SSH port for the openssh server
EXPOSE 22
#-------------VS code setup-------------------------------------

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# (Optional) Make python3 point to Python 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# install pip packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install PyYAML mkl-static mkl-include typing-extensions==4.12.2 matplotlib numpy 
RUN python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Clone a public Git repository
RUN git clone https://github.com/Shiweiliuiiiiiii/fp4llmt.git

# Optionally, switch to a specific commit, branch, or tag
# RUN cd fp4llmt && git checkout tags/2.3.3

RUN python3 -m pip install -e /fp4llmt[all]

#-------------- WIP ----------------------
# pip install other packages

# Install OlMO https://github.com/allenai/OLMo/blob/main/

# git cloning, needs Bas his FP4 kernal
# ls FP4_torch_kernel_main

# install FP4  kernel from Bas https://github.com/TianjinYellow/RealFP4/tree/master/FP4_torch_kernel_main
# RUN python /code/REALFP4.git/FP4_torch_kernel_main/setup.py develop
#-------------------------------------------

# Set the cloned repo as working directory
WORKDIR /fp4llmt

# Start SSH daemon
CMD ["/usr/sbin/sshd", "-D"]