import streamlit as st
import pandas as pd
from bertopic import BERTopic
from assets import innit
from assets import train_bertopic
from typing import Tuple, List
from assets.innit import format_labels, get_topic_index #, LIST_EMOTIONS
from assets.visualizations import create_wordcloud, plot_common_words
# number of buttons user can interact with per topic
N_BUTTONS = 4


innit.set_up()

# NOTE: THINGS TO CHECK :
# - st.form


# using copy bertopic visualization

# TO DO
# -- IF NOT CACHED image changes each time a diff topic is selected
#       check how to delete cache and generate only when pressed "view changes" button

# -- Add more options, e.g. reduce number topics automatically

# -- Interaction after deleting topic not smooth, check options

@st.cache_data
def generate_fig():
    return st.session_state.model.visualize_topics()

def update_fig():
    st.cache_data.clear()

def reset_buttons(number_buttons: int) -> List[bool]:
    st.session_state.clicked = [False] * number_buttons


def update_topic_names():
    st.session_state.model.set_topic_labels(st.session_state.model.topic_labels_)


# when doing merge or dropping topics need to update some variables in state_session, e.g. topic labels, emotions by topic
def update_state_session():
    update_topic_names()


option = st.selectbox(
    'Select topic:',
    format_labels(st.session_state['model'].topic_labels_),
    on_change=reset_buttons, args=[N_BUTTONS],
    key="selected_label"
)

if option:
    st.session_state.selected_topic = get_topic_index(option)

col1, col2 = st.columns([2, 2])

with col1:
    fig = generate_fig()
    st.plotly_chart(figure_or_data=fig, use_container_width=True)
    #st.plotly_chart(figure_or_data=st.session_state.figure, use_container_width=True)
    button_update_fig = st.button("View Changes", on_click=update_fig)

    #wordcloud most common words
    wordcloud = create_wordcloud(st.session_state['model'], topic=st.session_state.selected_topic)
    wordcloud.show()
    st.pyplot(wordcloud)



with col2:
    # Initialize the key in session state
    if 'clicked' not in st.session_state:
        # hmap to keep track in session of which buttons were pressed
        reset_buttons(N_BUTTONS)


    # Function to update the value in session state
    def clicked(button_index: int):
        # TO DO: Change representation button so only one can be set at a time e.g. just an int and don't need to be reset each time
        reset_buttons(N_BUTTONS)
        st.session_state.clicked[button_index] = True


    if option:
        st.session_state.selected_topic = get_topic_index(option)
        col2a, col2b, col2c, col2d = st.columns([2, 2, 2, 1])

        with col2a:
            b1 = st.button("Rename", on_click=clicked, args=[0])


            def save_name():

                if st.session_state['model'].topic_labels_[st.session_state.selected_topic] != st.session_state.new_name:
                    st.session_state['model'].topic_labels_[st.session_state.selected_topic] = st.session_state.new_name
                    update_topic_names()  # update custom labels
                st.session_state.clicked[0] = False

        with col2b:
            b2 = st.button("Merge", on_click=clicked, args=[1])

        with col2c:
            b3 = st.button("Delete", on_click=clicked, args=[2])

        with col2c:
            b3 = st.button("Save Changes", on_click=clicked, args=[3])

            def save_model():
                new_path = "models/" + st.session_state.new_model_name
                st.session_state.model.save(new_path)
                st.session_state.clicked[2] = False

        # pressed rename button
        if st.session_state.clicked[0]:
            new_name = st.text_input(
                "New Name:",
                value=st.session_state['model'].topic_labels_[st.session_state.selected_topic],
                placeholder=st.session_state.selected_label,
                label_visibility="collapsed",
                on_change=save_name, key="new_name"
            )

        # pressed merge button
        # if True:
        if st.session_state.clicked[1]:
            colm1, colm2 = st.columns([4, 1])


            def do_merge():

                if st.session_state.topics_to_merge and len(st.session_state.topics_to_merge) >= 2:
                    l = list(map(get_topic_index, st.session_state.topics_to_merge))
                    st.session_state['model'].merge_topics_no_docs(l)


                st.session_state.clicked[1] = False  # set merge button back to False in session state


            with colm1:
                options = st.multiselect(
                    label='Select topics to merge',
                    options=format_labels(st.session_state['model'].topic_labels_),
                    # default = st.session_state['model'].topic_labels_[st.session_state.selected_topic],
                    key='topics_to_merge'
                )

            with colm2:

                st.caption(" ")  # just to fix layout
                st.caption(" ")
                b_merge = st.button("Confirm", on_click=do_merge)

        # pressed drop button
        if st.session_state.clicked[2]:
            cold1, cold2 = st.columns([3, 2])

            def do_delete():
                time_warning = st.text(
                    "Dropping Topic, this will take a while"
                )
                l = [-1, st.session_state.selected_topic]
                print("dropping", l)
                st.session_state['model'].merge_topics_no_docs(l)

                st.session_state.clicked[2] = False

            with cold1:
                confirm_drop_message = st.text(
                    "Are you sure you want"
                )
                confirm_drop_message2 = st.text(
                    "to delete this topic?"
                )
            with cold2:
                b_delete = st.button("Confirm", on_click=do_delete)



        if st.session_state.clicked[3]:
            default_name = st.session_state.model_name + "_customized"
            new_model_name = st.text_input(
                "Save Model as:",
                value=default_name,
                placeholder=default_name,
                #label_visibility="collapsed",
                on_change=save_model, key="new_model_name"
            )

        # dataframe, display docs per selected topic
        df = pd.Series(st.session_state.model.get_representative_docs(st.session_state.selected_topic))
        st.dataframe(df, use_container_width=True)

        # update button for graph
        topic_number = st.session_state.selected_topic
        topic_name = st.session_state.model.topic_labels_[topic_number]

        st.download_button(
            "Download",
            df.to_csv(index=False).encode('utf-8'),
            topic_name + ".csv",
            "text/csv",
            key='download-csv'
        )



    barchart = plot_common_words(st.session_state['model'], st.session_state.selected_topic, 5)
    st.plotly_chart(figure_or_data=barchart, use_container_width=True)

