FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y virtualenv bash wget python3 python3-pip libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

COPY requirements.txt /srv/requirements.txt

RUN mkdir -p /srv/KaggleChallenge

########################################################
FROM builder AS setup

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
ENV PATH="/opt/conda/bin:$PATH"

RUN conda create -n myenv

RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN mkdir -p /srv/KaggleChallenge/src
RUN mkdir -p /srv/KaggleChallenge/mini_book
RUN mkdir -p /srv/KaggleChallenge/data
COPY src /srv/KaggleChallenge/src
COPY mini_book /srv/KaggleChallenge/mini_book
COPY data /srv/KaggleChallenge/data

WORKDIR /srv/KaggleChallenge

########################################################
FROM setup AS final_test

COPY --from=setup /opt/conda /opt/conda

ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN pip3 install -r /srv/requirements.txt
WORKDIR /srv/KaggleChallenge

RUN mypy src/kcu/*.py --ignore-missing-imports
RUN python3 -m pytest -v --junit-xml /srv/test_results.xml src/kcu/test.py
