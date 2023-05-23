FROM ubuntu:22.04

RUN apt-get update && apt-get install -y kafkacat python3-dev python3-pip sqlite3

RUN pip install jupyter ipykernel kafka-python requests streamlit scikit-learn pandas sqlalchemy==1.4.16 typer psycopg2-binary celery redis

RUN pip uninstall watchdog -y

WORKDIR /app
COPY ./ /app/

ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]