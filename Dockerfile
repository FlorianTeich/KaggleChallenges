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
ENV VIRTUAL_ENV "/opt/venv"
RUN virtualenv venv --python=python3.9 $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

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

COPY --from=setup /opt/venv /opt/venv

# activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /srv/KaggleChallenge
# run the entrypoint (only when the image is instantiated into a container)
RUN mypy src/kcu/*.py --ignore-missing-imports
RUN python3 -m pytest -v --junit-xml /srv/test_results.xml src/kcu/test.py
