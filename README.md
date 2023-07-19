# Interactive Topic Modeling

This project offers an Interactive Topic Modeling app built using [Streamlit](https://docs.streamlit.io) 
and [BERTopic](https://maartengr.github.io/BERTopic/index.html).



##  Set up

To get started, install the project by running:

`pip install -e .`


## Running the app

To run the app you simply need to *cd* to the project folder, then run the line:

`streamlit run Home.py`





## Structure of the app
There are four tabs on the app:

1. **Home**: Select the model you want to load. By default, the app will load the model `kaggle_data`, you can change that in `assets/innit` -> `DEFAULT_MODEL_NAME`
2. **Customize Topics**: See a representation of all topic clusters. Select a topic and see documents of that topic, \
change the name, merge topics, delete topics. Save changes made to the model.
3. **Full Table**: See (and download as *csv*) a table with all topic indexes, topic names, frequency and documents.
4. **Topic Similarity**: Create a heatmap of the similarity scores among all or selected topics. 


## Training new models

To train a new model, use the function `train_model` in `assets.train_model`.

- Train model from opensearch: see an example in `train_tm_from_data_loader.py`.
- Train model from csv: see an example in `train_tm_from_csv.py`.


The function `train_model` will save the model in the *models* folder by default, then you can rerun the
Streamlit app and will be able to interact with the model.

 