import streamlit as st
import pandas as pd
import numpy as np
import time

# Start timer
# start_time = time.perf_counter()

st.set_page_config(page_title="Analisi")


try:
    if st.query_params.lang == 'english':
        st.header("Data Analysis")
    else:
        st.header("Analisi dei Dati")
except:
    st.header("Analisi dei Dati")


#### GET DATA

@st.cache_data
def get_data():
    df = pd.DataFrame(np.random.randn(1000, 3), columns=["a", "b", "c"])
    return df

df = get_data()


# # End timer
# end_time = time.perf_counter()

# # Calculate elapsed time
# elapsed_time = end_time - start_time
# st.write("Elapsed time: ", elapsed_time)


st.bar_chart(df)
