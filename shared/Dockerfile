# 1. Base image (Debian Bullseye) where OpenJDK 11 is available
FROM --platform=linux/amd64 python:3.10-slim-bullseye

# 2. Install OpenJDK 11 and wget
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk-headless wget && \
    rm -rf /var/lib/apt/lists/*

# 3. Install Miniconda (Base Python 3.8)
RUN mkdir -p /opt/miniconda_install \
 && wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /opt/miniconda_install/miniconda.sh \
 && bash /opt/miniconda_install/miniconda.sh -b -p /opt/miniconda \
 && rm -rf /opt/miniconda_install

# 4. Set PATH for future runs
ENV PATH="/opt/miniconda/bin:$PATH"

# 5. Create new environment using Mamba (The Solver is built-in to Mamba-based installation!)
COPY trans-eval.yml /tmp/environment.yaml 
# Use a separate RUN command to install Mamba, then create the environment
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda install -y -c conda-forge mamba && \
    # Create the environment directly from the YAML, Mamba is fast!
    mamba env create -f /tmp/environment.yaml && \
    # Clean cache and index files specifically
    conda clean -t -i -y && \
    rm /tmp/environment.yaml

# 6. Set environment to use the NEW environment's path
ENV CONDA_DEFAULT_ENV="trans-eval"
ENV PATH="/opt/miniconda/envs/trans-eval/bin:$PATH"

# 7. Set up application directory
RUN mkdir /app 
WORKDIR /app
# 8. Copy all necessary application files
COPY . .