import os
from sqlalchemy import create_engine

INPUT_HOST = "postgres"
INPUT_DB_TABLE = "car_data"

OUTPUT_DB = "/app/results.db"
OUTPUT_DB_TABLE = "results"

def get_output_connection():
    return create_engine('postgresql://' + os.getenv("POSTGRES_USER") + ':' + os.getenv("POSTGRES_PASSWORD")  + '@' + INPUT_HOST + '/' + os.getenv("POSTGRES_DB") )

def get_input_connection():
    return create_engine('postgresql://' + os.getenv("POSTGRES_USER") + ':' + os.getenv("POSTGRES_PASSWORD")  + '@' + INPUT_HOST + '/' + os.getenv("POSTGRES_DB") )
