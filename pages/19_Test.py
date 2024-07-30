# # streamlit_app.py

# import streamlit as st
# from st_files_connection import FilesConnection

# # Create connection object and retrieve file contents.
# # Specify input format is a csv and to cache the result for 600 seconds.
# conn = st.connection('gcs', type=FilesConnection)
# df = conn.read("nusebacra_bucket1/SSIC Fact Ref Table.csv", input_format="csv", ttl=600)

# st.table(df)

import streamlit as st
import pandas as pd

# Sample data
data = {'Column1': ['A', 'B', 'C'], 'Column2': [1, 2, 3]}
df = pd.DataFrame(data)

# Function to manipulate the selected value
def manipulate_value(value):
    return f"Manipulated Value: {value * 2}"

# Display the table
st.write("Click on a row to select a value:")
st.dataframe(df)

# Dropdown to select a value from the table
selected_value = st.selectbox("Select a value from Column2:", df['Column2'])

if st.button('Process Selected Value'):
    result = manipulate_value(selected_value)
    st.write(result)