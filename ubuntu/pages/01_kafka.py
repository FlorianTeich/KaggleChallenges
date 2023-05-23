import datetime
import sys
sys.path.append("../")
import DummyFile
import ml_service
import pandas as pd
import streamlit as st


st.title("Kafka Streaming Demo")

st.write("This App demonstrates the power of Data Streaming. Streamlit will insert data that you create here into a Postgres Database. A Kafka-Connect workload monitors the Postgres and creates messages if anything changes. Messages get sent to the Kafka broker and down the line to the subscribers. A python script consumes the message and creates a celery task in order to evaluate the input data on a pretrained model. The results of the model will get written into a result-table. This event will trigger the app to refresh automatically. Enjoy!")

input_engine = ml_service.get_input_connection()
inputdata = pd.read_sql_table(ml_service.INPUT_DB_TABLE, input_engine)

output_engine = ml_service.get_output_connection()
outdata = pd.read_sql_table(ml_service.OUTPUT_DB_TABLE, output_engine)

st.header("Entries")
st.dataframe(inputdata)

st.header("Results")
st.dataframe(outdata)

st.header("Create new Entry")

number_a = st.number_input('feature a')
number_b = st.number_input('feature b')

if st.button('Store new Entry'):

    # Add new entry to input database
    temp_data = pd.DataFrame({
        "feature_a": [number_a],
        "feature_b": [number_b]
    })
    temp_data.to_sql(ml_service.INPUT_DB_TABLE, input_engine, index=False, if_exists="append")

    # Trigger reload to refresh the input database-view in the streamlit app
    dummy_path = DummyFile.__file__
    with open(dummy_path, "w", encoding="utf-8") as filehandler:
        filehandler.write(f'timestamp = "{datetime.datetime.now()}"')
