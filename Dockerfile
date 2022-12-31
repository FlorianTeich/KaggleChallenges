# BASE
FROM python:3.9.13-slim as base
ENV LANG C.UTF-8 \
    LC_ALL C.UTF-8 \
    PYTHONDONTWRITEBYTECODE 1 \
    PYTHONFAULTHANDLER 1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

# PYTHON-DEPS
FROM base AS python-deps
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.3.1
RUN pip install pipenv
RUN apt-get update && apt-get install -y gcc g++ gfortran libopenblas-dev liblapack-dev
RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt | /venv/bin/pip install -r /dev/stdin
COPY . .
RUN poetry build && /venv/bin/pip install dist/*.whl

# RUNTIME
FROM base AS runtime
RUN apt-get update && apt-get install -y default-jdk
RUN useradd --create-home appuser
RUN mkdir -p /srv/KaggleChallenge
COPY run_pytest.sh /srv/KaggleChallenge
RUN chmod +x /srv/KaggleChallenge/run_pytest.sh
RUN chown -R appuser:appuser /srv/KaggleChallenge
USER appuser
COPY --from=python-deps /venv /venv
ENV PATH="/venv/bin:$PATH"
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
CMD ["tail", "-F", "anything"]
