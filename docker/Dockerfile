ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"


FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install mmcv
ARG MMCV="1.7.1"
RUN pip install -U openmim && mim install mmcv-full==${MMCV}

# Verify the installation
RUN python -c 'import mmcv;print(mmcv.__version__)'

# Install MMSegmentation as a package
ENV FORCE_CUDA="1"
RUN pip install mmsegmentation==0.30.0

# # Install AerialFormer
# RUN git clone https://github.com/UARK-AICV/AerialFormer.git /AerialFormer
# WORKDIR /AerialFormer
# RUN pip install --no-cache-dir -e .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
