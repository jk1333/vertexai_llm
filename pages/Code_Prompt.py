import streamlit as st
from google.cloud import aiplatform
import vertexai
from vertexai.preview.language_models import CodeGenerationModel

SESSION_KEY = "codeprompt"
HISTORY_KEY = "codeprompt_hs"
st.set_page_config(
    page_title="Code Prompt",
    page_icon="⚙️",
    layout="wide", 
    initial_sidebar_state="auto"
)

BASE_MODELS = ["code-bison@001"]

@st.cache_resource
def StartCodeModels(basemodel_name):
    vertexai.init()
    basemodel = CodeGenerationModel.from_pretrained(basemodel_name)
    models = {}
    models[basemodel_name] = basemodel
    for tuned_model in basemodel.list_tuned_model_names():
        registry = aiplatform.Model(tuned_model)
        models[registry.display_name] = CodeGenerationModel.get_tuned_model(tuned_model)
    return models

with st.sidebar:
    basemodel_name = st.selectbox("Base Models", BASE_MODELS)

models = StartCodeModels(basemodel_name)
parameters = {}
with st.sidebar:
    selected_model = st.selectbox("Tuned Models", models.keys())
    prompt = models[selected_model]
    expander = st.expander("Parameters")
    parameters['temperature'] = expander.slider("temperature", 0.0, 1.0, 0.2)
    parameters['max_output_tokens'] = expander.slider("Max output tokens", 1, 2048, 1024)

st.title("⚙️ Code prompt")
if st.button("♻️"):
    del st.session_state[SESSION_KEY]
    st.cache_resource.clear()
    st.experimental_rerun()

if SESSION_KEY not in st.session_state:
    st.session_state[SESSION_KEY] = {"request": "", "response": None}
    st.session_state[HISTORY_KEY] = []

def textChange():
    st.session_state[SESSION_KEY]["request"] = st.session_state.input
st.text_area("Enter Task:", st.session_state[SESSION_KEY]["request"], key='input', on_change=textChange)
request = st.session_state[SESSION_KEY]["request"]
response = st.session_state[SESSION_KEY]["response"]

if st.button("Execute"):
    with st.spinner("Thinking..."):
        response = prompt.predict(request, **parameters)
    st.session_state[SESSION_KEY]["response"] = response
    st.session_state[HISTORY_KEY].append({"request": request, "response": response})

if response != None:
    st.code(response, line_numbers=True)
    st.write("Safety attributes")
    if response.is_blocked:
        st.error(response.safety_attributes, icon="🚨")
    else:
        st.success(response.safety_attributes, icon="✅")
        citations = response._prediction_response.predictions[0]['citationMetadata']['citations']
        if len(citations) > 0:
            st.write("Citations")
            out = ""
            for citation in citations:
                out += str(citation) + "\n\r"
            st.warning(out, icon="⚠️")

with st.sidebar:
    expander = st.expander("History")
    if expander.button("Clear history", type="primary", use_container_width=True):
        del st.session_state[HISTORY_KEY]
        st.session_state[HISTORY_KEY] = []
    for idx, history in reversed(list(enumerate(st.session_state[HISTORY_KEY]))):
        if expander.button(f'{idx+1}.{history["request"]}', use_container_width=True):
            st.session_state[SESSION_KEY]["request"] = history["request"]
            st.session_state[SESSION_KEY]["response"] = history["response"]
            st.experimental_rerun()