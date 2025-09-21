FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /workspace

COPY conda.yml .

RUN conda env create -f conda.yml

RUN wget https://ailb-web.ing.unimore.it/publicfiles/drive/show-control-and-tell/flickr30k_detections.hdf5 && \
    wget https://ailb-web.ing.unimore.it/publicfiles/drive/show-control-and-tell/saved_models.tgz && \
    wget https://ailb-web.ing.unimore.it/publicfiles/drive/show-control-and-tell/dataset_flickr.tgz

RUN tar xzf saved_models.tgz && \
    tar xzf dataset_flickr.tgz && \
    rm saved_models.tgz dataset_flickr.tgz

SHELL ["conda", "run", "-n", "sct", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "-n", "sct"]
CMD ["/bin/bash"]