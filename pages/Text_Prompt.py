import streamlit as st
import vertexai
from vertexai.language_models import TextGenerationModel

SESSION_KEY = "textprompt"
HISTORY_KEY = "textprompt_hs"
st.set_page_config(
    page_title="Text Prompt",
    page_icon="📝",
    layout="wide", 
    initial_sidebar_state="auto"
)

parameters = {}
with st.sidebar:
    expander = st.expander("Parameters")
    parameters['temperature'] = expander.slider("Temperature", 0.0, 1.0, 0.2)
    parameters['max_output_tokens'] = expander.slider("Max output tokens", 1, 1024, 256)
    parameters['top_k'] = expander.slider("Top K", 1, 40, 40)
    parameters['top_p'] = expander.slider("Top P", 0.0, 1.0, 0.8)

@st.cache_resource
def StartTextPrompt():
    vertexai.init()
    return TextGenerationModel.from_pretrained("text-bison")

prompt = StartTextPrompt()

st.title("📝 Text prompt")
if st.button("♻️"):
    del st.session_state[SESSION_KEY]

if SESSION_KEY not in st.session_state:
    st.session_state[SESSION_KEY] = {"request": "", "response": None}
    st.session_state[HISTORY_KEY] = []

request = st.text_area("Enter Task:", st.session_state[SESSION_KEY]["request"])
st.session_state[SESSION_KEY]["request"] = request
response = st.session_state[SESSION_KEY]["response"]

if st.button("Execute"):
    with st.spinner("Thinking..."):
        response = prompt.predict(request, **parameters)
    st.session_state[SESSION_KEY]["response"] = response
    st.session_state[HISTORY_KEY].append({"request": request, "response": response})

if response != None:
    st.info(response)
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