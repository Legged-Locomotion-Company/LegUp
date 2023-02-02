FROM nvcr.io/nvidia/pytorch:21.09-py3
ENV DEBIAN_FRONTEND=noninteractive 

# dependencies for gym
#
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    make \
    gcc-8 \
    g++-8 \
    vulkan-utils \
    mesa-vulkan-drivers \
    pigz \
    git \
    libegl1 \
    git-lfs \
    megatools \
    ssh

# # Install python dependencies
# RUN pip install \
#     opencv-python==4.5.4.60 \
#     colorama \
#     gym \
#     numpy \
#     opencv_python \
#     pytest \
#     pytorch3d \
#     scipy \
#     setuptools \
#     stable_baselines3 \
#     torchgeometry \
#     moviepy \
#     imageio \
#     wandb \
#     hydra-core \
#     uuid

WORKDIR /opt/LegUp

RUN megadl 'https://mega.co.nz/#!6f4jnDqB!b_z5kvu8yfmmRdvxfAbSgXa69QSAOcWlkiMCEFTxJ6M' \
    && tar -xvzf IsaacGym_Preview_4_Package.tar.gz

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

RUN cp ./isaacgym/docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
RUN cp ./isaacgym/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN useradd --create-home legsuser
RUN chown -R legsuser:legsuser /opt/LegUp
USER legsuser

# install gym modules
ENV PATH="/home/legsuser/.local/bin:$PATH"
RUN pip install -q -e ./isaacgym/python/.

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

ENTRYPOINT ["sh"]