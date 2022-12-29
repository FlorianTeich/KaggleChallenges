# BASE
FROM python:3.10.7-slim as base
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

# PYTHON-DEPS
FROM base AS python-deps
RUN pip install pipenv
RUN apt-get update && apt-get install -y gcc g++ gfortran libopenblas-dev liblapack-dev
RUN python -m venv /venv
COPY Pipfile Pipfile.lock ./
ENV PIPENV_VENV_IN_PROJECT 1
RUN pipenv install

# RUNTIME
FROM base AS runtime
RUN apt-get update && apt-get install -y default-jdk
RUN useradd --create-home appuser
RUN mkdir -p /srv/KaggleChallenge
COPY run_pytest.sh /srv/KaggleChallenge
RUN chmod +x /srv/KaggleChallenge/run_pytest.sh
RUN chown -R appuser:appuser /srv/KaggleChallenge
USER appuser
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"
COPY . .
RUN mkdir -p /srv/KaggleChallenge/src && \
    mkdir -p /srv/KaggleChallenge/mini_book && \
    mkdir -p /srv/KaggleChallenge/data && \
    mkdir -p /srv/KaggleChallenge/extra_data && \
    mkdir -p /srv/KaggleChallenge/plugins
COPY src /srv/KaggleChallenge/src
COPY mini_book /srv/KaggleChallenge/mini_book
COPY data /srv/KaggleChallenge/data
COPY plugins /srv/KaggleChallenge/plugins
WORKDIR /srv/KaggleChallenge
#RUN chown -R appuser:appuser /srv/KaggleChallenge
#COPY --from=python-deps /Pipfile.lock /srv/KaggleChallenge/Pipfile.lock
CMD ["tail", "-F", "anything"]
