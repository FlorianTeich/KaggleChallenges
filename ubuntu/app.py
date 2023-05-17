import datetime
import DummyFile
import ml_service
import pandas as pd
import streamlit as st


st.title("Kafka Streaming Demo")

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
