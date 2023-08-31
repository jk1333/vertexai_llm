import streamlit as st
from google.cloud import aiplatform
import vertexai
from vertexai.preview.language_models import ChatModel

SESSION_KEY = "textchat"
USER_ICON = "😊"
BOT_ICON = "🤖"
st.set_page_config(
    page_title="Text Chat",
    page_icon="💬",
    layout="wide", 
    initial_sidebar_state="auto"
)

BASE_MODELS = ["chat-bison@001"]

@st.cache_resource
def StartTextChatModels(basemodel_name):
    vertexai.init()
    basemodel = ChatModel.from_pretrained(basemodel_name)
    models = {}
    models[basemodel_name] = basemodel
    for tuned_model in basemodel.list_tuned_model_names():
        registry = aiplatform.Model(tuned_model)
        models[registry.display_name] = ChatModel.get_tuned_model(tuned_model)
    return models

with st.sidebar:
    basemodel_name = st.selectbox("Base Model", BASE_MODELS)

models = StartTextChatModels(basemodel_name)

@st.cache_resource
def StartChat(selected_model):
    return models[selected_model].start_chat()

parameters = {}
with st.sidebar:
    selected_model = st.selectbox("Tuned Models", models.keys())
    chat = StartChat(selected_model)
    expander = st.expander("Parameters")
    parameters['temperature'] = expander.slider("temperature", 0.0, 1.0, 0.2)
    parameters['max_output_tokens'] = expander.slider("Max output tokens", 1, 1024, 256)
    parameters['top_k'] = expander.slider("Top K", 1, 40, 40)
    parameters['top_p'] = expander.slider("Top P", 0.0, 1.0, 0.8)

st.title("📝💬 Chatbot")
if st.button("♻️"):
    del st.session_state[SESSION_KEY]
    st.cache_resource.clear()
    st.experimental_rerun()

if SESSION_KEY not in st.session_state:
    st.session_state[SESSION_KEY] = [{"role": "assistant", "icon": BOT_ICON, "content": "How can I help you?"}]

for msg in st.session_state[SESSION_KEY]:
    st.chat_message(msg["role"], avatar=msg["icon"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state[SESSION_KEY].append({"role": "user", "icon": USER_ICON, "content": prompt})
    st.chat_message("user", avatar=USER_ICON).write(prompt)
    with st.spinner("Thinking..."):
        response = chat.send_message(st.session_state[SESSION_KEY][-1]['content'], **parameters)
    st.session_state[SESSION_KEY].append({"role": "assistant", "icon": BOT_ICON, "content": response.text})
    st.chat_message("assistant", avatar=BOT_ICON).write(response.text)