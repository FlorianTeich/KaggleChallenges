import os
import time
import json
import kafka
import typer
import sqlite3
import requests
import psycopg2
import datetime
import DummyFile
import numpy as np
import pandas as pd
from celery import Celery
from joblib import dump, load
from kafka import KafkaConsumer
from sqlalchemy import create_engine
from sklearn.neural_network import MLPClassifier
import ml_celery
from config import *

app = typer.Typer()

@app.command()
def register_connector():
    registered = False
    while registered == False:
        time.sleep(1)
        try:
            print("Sending reqest...")
            result = requests.post(
                "http://connect:8083/connectors",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                "name": "source_pg_car_connector",
                "config": {
                    "connector.class":"io.debezium.connector.postgresql.PostgresConnector",
                    "plugin.name": "pgoutput",

                    "database.hostname": "postgres",
                    "database.port": "5432",
                    "database.user": os.getenv("POSTGRES_USER"),
                    "database.password": os.getenv("POSTGRES_PASSWORD"),
                    "database.dbname": os.getenv("POSTGRES_DB"),

                    "database.server.name": "car_database",
                    "table.include.list": "public.car_data",
                    "topic.prefix": "pref",

                    "transforms": "unwrap",
                    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
                    }
                }
                )
            print("Request Response:", result.status_code)
            if result.status_code == 409:
                registered = True
        except:
            pass
        print("Connection successfully registered!")

@app.command()
def reset_output_db():
    print("Creating SQL-Table...")
    #sqlite3.connect(OUTPUT_DB)
    engine = get_output_connection()
    data = pd.DataFrame({
        "feature_a": pd.Series(dtype='float'),
        "feature_b": pd.Series(dtype='float'),
        "score": pd.Series(dtype='int')
    })
    data.to_sql(OUTPUT_DB_TABLE, engine, if_exists="replace")
    print("Created SQL-Table successfully!")

@app.command()
def prepare_ml_service():
    model = MLPClassifier(alpha=1, max_iter=1000)
    X_train = np.random.randn(100, 2)
    y_train = np.random.randint(0, 1, (100,))
    model.fit(X_train, y_train)
    dump(model, '/app/model.joblib')

@app.command()
def run_ml_service():
    consumer = KafkaConsumer('pref.public.car_data',
        bootstrap_servers=["kafka:9092"],
        group_id=None,
        auto_offset_reset='earliest')

    for msg in consumer:
        print(msg)
        payload = json.loads(msg.value)["payload"]
        ml_celery.run_inference.delay(payload)

@app.command()
def run_ml_service_blocking():
    model = load('/app/model.joblib')

    consumer = KafkaConsumer('pref.public.car_data',
        bootstrap_servers=["kafka:9092"],
        group_id=None,
        auto_offset_reset='earliest')

    for msg in consumer:
        print(msg)
        payload = json.loads(msg.value)["payload"]
        result = model.predict(np.array([[payload["feature_a"], payload["feature_b"]]]))
        data_new = pd.DataFrame({
            "feature_a": [payload["feature_a"]],
            "feature_b": [payload["feature_b"]],
            "score": [result.item()]
        })
        data_new.to_sql(OUTPUT_DB_TABLE, get_output_connection(), if_exists="append")
        dummy_path = DummyFile.__file__
        with open(dummy_path, "w", encoding="utf-8") as filehandler:
            filehandler.write(f'timestamp = "{datetime.datetime.now()}"')

if __name__ == "__main__":
    app()
