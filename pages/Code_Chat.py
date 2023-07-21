import streamlit as st
import vertexai
from vertexai.preview.language_models import CodeChatModel

SESSION_KEY = "codechat"
USER_ICON = "😊"
BOT_ICON = "🤖"
st.set_page_config(
    page_title="Code Chat",
    page_icon="⚙️",
    layout="wide", 
    initial_sidebar_state="auto"
)

parameters = {}
with st.sidebar:
    expander = st.expander("Parameters")
    parameters['temperature'] = expander.slider("temperature", 0.0, 1.0, 0.2)
    parameters['max_output_tokens'] = expander.slider("Max output tokens", 1, 2048, 1024)

@st.cache_resource
def StartCodeChat():
    vertexai.init()
    chat_model = CodeChatModel.from_pretrained("codechat-bison")
    return chat_model.start_chat()

chat = StartCodeChat()

st.title("⚙️💬 CodeChatbot")
if st.button("♻️"):
    del st.session_state[SESSION_KEY]

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