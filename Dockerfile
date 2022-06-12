FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# install the prerequisites
RUN apt-get update && apt-get install -y virtualenv python3 python3-pip libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

# copy in the requirements
COPY requirements.txt /srv/requirements.txt

# create a folder to hold the downloaded/built requirements
RUN mkdir -p /srv/KaggleChallenge

FROM builder AS setup

# Set up and activate virtual environment
# https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
# ENV PATH "$VIRTUAL_ENV/bin:$PATH"

# install the requirements to that folder
RUN pip3 install -r /srv/requirements.txt --target /srv/KaggleChallenge

# copy in the code to be tested (this merges with the folder above)
RUN mkdir -p /srv/KaggleChallenge/src
RUN mkdir -p /srv/KaggleChallenge/mini_book
RUN mkdir -p /srv/KaggleChallenge/data
COPY src /srv/KaggleChallenge/src
COPY mini_book /srv/KaggleChallenge/mini_book
COPY data /srv/KaggleChallenge/data

# change to the appropriate folder
WORKDIR /srv/KaggleChallenge

FROM setup AS final_test

COPY --from=setup /opt/conda /opt/conda

# activate virtual environment
# ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/conda/bin:$PATH"
RUN conda activate

WORKDIR /srv/KaggleChallenge
# run the entrypoint (only when the image is instantiated into a container)
RUN mypy src/kcu/*.py --ignore-missing-imports
RUN python3 -m pytest -v --junit-xml /srv/test_results.xml src/kcu/test.py
