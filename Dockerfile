FROM python:3.10-slim as base

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1


FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy


FROM base AS runtime

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

# create a folder to hold the downloaded/built requirements
RUN mkdir -p /srv/KaggleChallenge

# copy in the code to be tested (this merges with the folder above)
RUN mkdir -p /srv/KaggleChallenge/src
RUN mkdir -p /srv/KaggleChallenge/mini_book
RUN mkdir -p /srv/KaggleChallenge/data
COPY src /srv/KaggleChallenge/src
COPY mini_book /srv/KaggleChallenge/mini_book
COPY data /srv/KaggleChallenge/data

# change to the appropriate folder
WORKDIR /srv/KaggleChallenge

# run the entrypoint (only when the image is instantiated into a container)
RUN python -m pytest -v --junit-xml /srv/test_results.xml src/kcu/test.py
