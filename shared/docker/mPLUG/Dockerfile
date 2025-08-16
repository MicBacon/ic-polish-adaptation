FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3.9 python3.9-venv python3.9-dev python3-pip git nano wget unzip untar \
	software-properties-common build-essential python3-setuptools make cmake \
	ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev && \
	apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


# Ustawienie domyślnego Pythona i PIP
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
	ln -sf $(which pip3) /usr/bin/pip

# Ustawienie katalogu roboczego
WORKDIR /workspace/main

# Kopiowanie plików projektu
COPY . .

# Instalacja zależności Pythona
RUN python -m pip install --upgrade pip
RUN python -m pip install -r mPLUG/requirements.txt

# Instalacja bibliotek ze źródeł
RUN git clone --recursive https://github.com/dmlc/decord && \
	cd decord && \
	mkdir build && cd build && \
	cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release && \
	make

RUN git clone https://github.com/NVIDIA/apex && \
	cd apex && \
	# Fix Python 3.9 compatibility
	sed -i '1i from typing import Optional' setup.py && \
	sed -i 's/parallel: int | None = None/parallel: Optional[int] = None/' setup.py && \
	# Disable CUDA version check
	sed -i '/def check_cuda_torch_binary_vs_bare_metal(cuda_dir):/,/^$/s/^/#/' setup.py && \
	sed -i '/check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/s/^/#/' setup.py && \
	# Set environment variables
	export CUDA_HOME=/usr/local/cuda && \
	export TORCH_CUDA_ARCH_LIST="9.0" && \
	export MAX_JOBS=4 && \
	# Install
	python setup.py install --cuda_ext --cpp_ext && \
	# Apply compatibility patches
	SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && \
	find "$SITE_PACKAGES/apex" -type f -name "*.py" -exec sed -i \
	-e "s/from torch\._six import container_abcs/import collections.abc as container_abcs/g" \
	-e "s/from torch\._six import string_classes/from torch._utils import string_classes/g" \
	-e "s/import torch\._six/import collections.abc as container_abcs/g" {} +

RUN cd /workspace/main/ic-polish-adaptation/mPLUG && \
	git clone https://github.com/j-min/language-evaluation && \
	pip install -e language-evaluation

# pobranie datasetu coco
RUN cd /workspace/main/ic-polish-adaptation/shared/data/coco_object/ && \
	wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/data.tar && \
	untar -xvf data.tar -C /workspace/main/ic-polish-adaptation/shared/data/coco_object/ && \
	rm data.tar && cd data && mv -v coco_object/.[!.]* /workspace/main/ic-polish-adaptation/shared/data/coco_object/ && \
	rm -rf data && cd coco_2014 && mkdir val_2014 train_2014 test_2014 && cd train2014 && wget http://images.cocodataset.org/zips/train2014.zip && \
	unzip train2014.zip && rm train2014.zip && cd ../val_2014 && wget http://images.cocodataset.org/zips/val2014.zip && \
	unzip val2014.zip && rm val2014.zip && cd ../test_2014 && wget http://images.cocodataset.org/zips/test2014.zip && \


	# pobranie checkpointów oraz backbone
	RUN cd /workspace/main/mPLUG && \
	wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/mplug_large_v2.pth && \
	wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-L-14.tar
