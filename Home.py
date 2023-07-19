import streamlit as st
from assets import train_bertopic, innit
from os import path
from bertopic import BERTopic
import os

st.set_page_config(
    page_title="",
    layout="wide"
)


innit.set_up()

st.write("# Welcome! 👋")

st.write("# Choose a model:")

model_options =st.selectbox(
        label='Select a model to visualize',
        options=os.listdir('models'),
        key='model_name'
        )

st.session_state.model = BERTopic.load("models/"+st.session_state.model_name)

st.caption(" ")
st.caption(" ")

st.markdown(
    """
    This is an interactive topic modeling tool built using [streamlit.io](https://streamlit.io)
    and [BERTopic](https://maartengr.github.io/BERTopic/index.html).
    """)

st.markdown(
    "**👈 Select a page from the sidebar!**"
)
#     ### Want to learn more about streamlit?
#     - Check out [streamlit.io](https://streamlit.io)
#     - Jump into our [documentation](https://docs.streamlit.io)
#     - Ask a question in our [community
#         forums](https://discuss.streamlit.io)
#     ### See more about BERTopic
#     - Check out
# """
# )
