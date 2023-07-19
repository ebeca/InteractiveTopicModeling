import streamlit as st
from assets import innit
import pandas as pd

# TO DO: st.experimental_data_editor

innit.set_up()

# all_docs = train_bertopic.get_all_docs(st.session_state.model, st.session_state.docs)

all_docs = pd.DataFrame.from_dict(st.session_state.model.get_representative_docs(), orient="index")

all_docs.insert(0, "Topic Name", st.session_state.model.topic_labels_.values())
all_docs.insert(1, "Topic Frequency", st.session_state.model.topic_sizes_.values())

st.dataframe(all_docs, use_container_width=True)


st.download_button(
        "Download",
        all_docs.to_csv(index=False).encode('utf-8'),
        "all_documents_per_topic.csv",
        "text/csv",
        key='download-all-csv'
    )
