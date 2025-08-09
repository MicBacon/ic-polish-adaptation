FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# Ustawienie DEBIAN_FRONTEND na noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3.9 python3.9-venv python3.9-dev python3-pip git nano wget unzip \
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

# RUN cd /workspace/main/mPLUG && \
# 	git clone https://github.com/j-min/language-evaluation && \
# 	pip install -e language-evaluation

