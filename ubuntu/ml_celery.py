import os
import time
import datetime
import psycopg2
from config import *
import DummyFile
import numpy as np
import pandas as pd
from celery import Celery
from joblib import dump, load
from sqlalchemy import create_engine

celapp = Celery(
   main='myproject',
   broker='redis://redis:6379',
   )

celapp.conf.task_track_started = True
celapp.conf.task_send_sent_event = True

celapp.autodiscover_tasks()

@celapp.task
def run_inference(payload):
    model = load('/app/model.joblib')
    time.sleep(5)
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
