import streamlit as st
import vertexai
from vertexai.preview.vision_models import Image, ImageCaptioningModel, ImageQnAModel

SESSION_KEY = "imageprompt"
st.set_page_config(
    page_title="Image Prompt",
    page_icon="🖼️",
    layout="wide", 
    initial_sidebar_state="auto"
)

with st.sidebar:
    expander = st.expander("Parameters")
    number_of_results = expander.slider("Num of results", 1, 3, 1)
    lang = expander.selectbox("Set output language", ('en', 'fr', 'de', 'it', 'es'))

@st.cache_resource
def StartImageModels():
    vertexai.init()
    return ImageCaptioningModel.from_pretrained("imagetext@001"), ImageQnAModel.from_pretrained("imagetext@001")

caption, prompt = StartImageModels()

st.title("🖼️ Image prompt")

if SESSION_KEY not in st.session_state:
    st.session_state[SESSION_KEY] = {"id": -1, "captions": None, "question": None, "answers": None}

uploaded_file = st.file_uploader("Choose a image", ['png', 'jpg'])
if uploaded_file is None:
    st.stop()

if st.session_state[SESSION_KEY]["id"] != uploaded_file.id:
    st.session_state[SESSION_KEY]["id"] = uploaded_file.id
    st.session_state[SESSION_KEY]["captions"] = None
    st.session_state[SESSION_KEY]["question"] = None
    st.session_state[SESSION_KEY]["answers"] = None

col = st.columns(3)[1]
bytes_data = uploaded_file.getvalue()
col.image(bytes_data, "Source", 400)

if st.session_state[SESSION_KEY]["captions"] == None:
    with st.spinner('Thinking...'):
        st.session_state[SESSION_KEY]["captions"] = caption.get_captions(image=Image(bytes_data), number_of_results=number_of_results, language=lang)

for cap in st.session_state[SESSION_KEY]["captions"]:
    st.info(cap)

question = st.text_input("Ask about image")

if len(question) == 0:
    st.stop()

if st.session_state[SESSION_KEY]["question"] != question:
    st.session_state[SESSION_KEY]["question"] = question
    st.session_state[SESSION_KEY]["answers"] = None

if st.session_state[SESSION_KEY]["answers"] == None:
    with st.spinner('Thinking...'):
        st.session_state[SESSION_KEY]["answers"] = prompt.ask_question(image=Image(bytes_data), question=question, number_of_results=number_of_results)

for answer in st.session_state[SESSION_KEY]["answers"]:
    st.info(answer)