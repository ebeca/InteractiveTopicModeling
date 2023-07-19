from os import path
import streamlit as st
from bertopic import BERTopic
import pandas as pd
from assets import train_bertopic
from typing import Tuple
from streamlit_extras.app_logo import add_logo

DEFAULT_MODEL_NAME = "kaggle_data"

HEIGHT_LOGO = 300
LOGO_PATH = "gallery/quandago.png"

def set_up():
    set_up_logo(LOGO_PATH, HEIGHT_LOGO)
    # setup_docs()
    # setup_model()
    # setup_emotions()
    if 'model' not in st.session_state:
        st.session_state.model = BERTopic.load("models/"+DEFAULT_MODEL_NAME)
        st.session_state.model_name = DEFAULT_MODEL_NAME


def set_up_logo(image_path: str, height: int):
    add_logo(image_path, height=height)


# TO DO: implement loading different topics
# # st.file_uploader

def setup_model():
    if 'model' not in st.session_state:
        # if not load model if there is one or train one
        if path.exists("models/" + DEFAULT_MODEL_NAME):
            print("Model found, loading model")
            st.session_state['model'] = BERTopic.load("models/" + DEFAULT_MODEL_NAME)
            st.session_state.model_name = DEFAULT_MODEL_NAME
        else:
            print("Model not found, train model first")




def load_text_data(file_path: str, column: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data[column]


def format_labels(hmap: dict) -> Tuple[str]:
    l = []
    for k in hmap:
        if k != -1:  # use -1 as the meaningless group
            l.append(f"{k}: {hmap[k]}")
    return tuple(l)


# TO DO: redefine (dict?)
def get_topic_index(name_of_topic: str) -> int:
    return int(name_of_topic.split(":")[0])


def cut_labels(hmap: dict) -> dict:
    short_labels = {}
    for k in hmap:
        original_label = hmap[k]
        split_label = original_label.split("_")
        short_label = "".join(split_label[0:min(2, len(split_label))])
        short_labels[k] = short_label
    return short_labels


