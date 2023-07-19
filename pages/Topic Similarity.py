import streamlit as st
from assets import innit
from assets.innit import format_labels, get_topic_index, cut_labels

N_BUTTONS = 2

innit.set_up()

# TO DO:
# --if there are custom labels, full label name appers (less readable graph, cut names)??
#       check how that is dealt with in bertopic visualization
# -- add more interaction, e.g. visualize top topics


if 'clicked_select' not in st.session_state:
    st.session_state.clicked_select = False


if 'comparison_fig' not in st.session_state:
    st.session_state.comparison_fig = st.session_state.model.visualize_heatmap(custom_labels=cut_labels(st.session_state.model.topic_labels_))


if 'model' not in st.session_state:
    innit.setup_model()

def clicked_all():
    st.session_state.clicked_select = False
    st.session_state.comparison_fig = st.session_state.model.visualize_heatmap(custom_labels=st.session_state.model.topic_labels_)
def clicked_compare():
    st.session_state.clicked_select= True

colbutton1, colbutton2, dummy_col = st.columns([2, 2, 4])

with colbutton1:
    b_similarity = st.button("Compare Topics", on_click=clicked_compare)
with colbutton2:
    b_all = st.button("Compare All", on_click=clicked_all)

# clicked button to compare specific topics
if st.session_state.clicked_select:

    col1, col2 = st.columns([4, 1])

    def compare():

        if st.session_state.topics_to_compare and len(st.session_state.topics_to_compare) >= 2:
            l = list(map(get_topic_index, st.session_state.topics_to_compare))
            st.session_state.comparison_fig = st.session_state.model.visualize_heatmap(topics=l, custom_labels=True)


    with col1:
        options = st.multiselect(
            label='Select topics to compare',
            options=format_labels(st.session_state['model'].topic_labels_),
            key='topics_to_compare'
        )

    with col2:

        st.caption(" ")  # just to fix layout
        st.caption(" ")
        b_merge = st.button("Confirm", on_click=compare)


st.plotly_chart(figure_or_data=st.session_state.comparison_fig, use_container_width=True)


