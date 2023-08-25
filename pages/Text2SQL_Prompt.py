import streamlit as st
import vertexai
from google.cloud.aiplatform.models import Endpoint
import sys
from google.cloud import bigquery
from streamlit_ace import st_ace

SESSION_KEY = "text2sqlprompt"
HISTORY_KEY = "text2sqlprompt_hs"
LOCATION = "us-west2"
PROJECT_ID = sys.argv[1]
ENDPOINT_ID = sys.argv[2]
st.set_page_config(
    page_title="Text2SQL Prompt",
    page_icon="📝",
    layout="wide", 
    initial_sidebar_state="auto"
)

@st.cache_resource
def StartBigqueryClient():
    return bigquery.Client()

bqclient = StartBigqueryClient()

with st.sidebar:
    datasets = []
    for dataset in bqclient.list_datasets():
        datasets.append(dataset.dataset_id)
    dataset = st.selectbox("Dataset to work", datasets)

@st.cache_resource
def StartText2SQLPrompt():
    vertexai.init()
    return Endpoint(endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}")

prompt = StartText2SQLPrompt()

def predict(nlquery, dataset, tables_filter = None, columns_filter = None):
    example_input = [
        {
            "text": nlquery,
            "bq_dataset": f"{PROJECT_ID}.{dataset}",
            "project_id": PROJECT_ID,
        }]
    if tables_filter:
        example_input[0]["bq_tables_filter"] = tables_filter
    if columns_filter:
        example_input[0]["bq_columns_filter"] = columns_filter

    params = {
        "temperature": 0.8,
        "nb_samples": 5,
    }
    return prompt.predict(instances=example_input, parameters=params).predictions[0]['sql_query']

st.title("📝 Text2SQL prompt")
if st.button("♻️"):
    del st.session_state[SESSION_KEY]

if SESSION_KEY not in st.session_state:
    st.session_state[SESSION_KEY] = {"request": "", "response": None, "dataframe": None}
    st.session_state[HISTORY_KEY] = []

request = st.text_area("Enter Task:", st.session_state[SESSION_KEY]["request"])
st.session_state[SESSION_KEY]["request"] = request
response = st.session_state[SESSION_KEY]["response"]

if st.button("Execute"):
    with st.spinner("Thinking..."):
        response = predict(request, dataset)

if response != None:
    response = st_ace(value=response, language="sql", theme="clouds", wrap=True)
    if st.session_state[SESSION_KEY]["response"] != response:
        df = None
        with st.spinner("Executing bigquery..."):
            try:
                df = bqclient.query(response).to_dataframe()
            except Exception as e:
                st.error(e)
        st.session_state[SESSION_KEY]["response"] = response
        st.session_state[SESSION_KEY]["dataframe"] = df
        st.session_state[HISTORY_KEY].append({"request": request, "response": response, "dataframe": df})
    st.dataframe(st.session_state[SESSION_KEY]["dataframe"])

with st.sidebar:
    expander = st.expander("History")
    if expander.button("Clear history", type="primary", use_container_width=True):
        del st.session_state[HISTORY_KEY]
        st.session_state[HISTORY_KEY] = []
    for idx, history in reversed(list(enumerate(st.session_state[HISTORY_KEY]))):
        if expander.button(f'{idx+1}.{history["request"]}', use_container_width=True):
            st.session_state[SESSION_KEY]["request"] = history["request"]
            st.session_state[SESSION_KEY]["response"] = history["response"]
            st.session_state[SESSION_KEY]["dataframe"] = history["dataframe"]
            st.experimental_rerun()