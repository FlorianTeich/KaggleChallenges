FROM python:3.10.7-slim as base

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1


FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv
RUN apt-get update && apt-get install -y gcc g++ gfortran libopenblas-dev liblapack-dev default-jdk

# Install python dependencies in /.venv
#RUN pip install "poetry==1.1.14"
RUN python -m venv /venv
#COPY pyproject.toml poetry.lock ./
COPY Pipfile ./
COPY Pipfile.lock ./
#RUN . /venv/bin/activate && pipenv install
ENV PIPENV_VENV_IN_PROJECT 1
RUN pipenv install
#RUN . /venv/bin/activate && poetry install --no-dev --no-root
#RUN . /venv/bin/activate && poetry build

FROM base AS runtime

# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"
COPY . .
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
#WORKDIR /srv/KaggleChallenge

# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /srv/KaggleChallenge
COPY run_pytest.sh ./run_pytest.sh
RUN chmod +x /srv/KaggleChallenge/run_pytest.sh
USER appuser

CMD ["tail", "-F", "anything"]
