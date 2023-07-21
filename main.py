import streamlit as st

st.set_page_config(
    page_title="LLM Bots",
    page_icon="🤖",
    layout="wide", 
    initial_sidebar_state="auto"
)

st.write("# LLM Bots Portal")

st.write("### Code prompt")
st.write("- Max input token: 4096")
st.write("- Max output token: 2048")

st.write("### Code chat")
st.write("- Max input token: 4096")
st.write("- Max output token: 2048")

st.write("### Text prompt")
st.write("- Max input token: 8192")
st.write("- Max output token: 1024")

st.write("### Text chat")
st.write("- Max input token: 4096")
st.write("- Max output token: 1024")