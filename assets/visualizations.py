import streamlit as st
from assets import train_bertopic, innit
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
import plotly.graph_objects as go


# wordcloud of most common words per topic
def create_wordcloud(model: BERTopic, topic: int):
    fig = plt.figure()
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    return fig



def plot_common_words(topic_model: BERTopic, topic: int, n_words: int):
    words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
    scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

    # Note: this only works in n_words = 5
    colors = ["#004c6d", "#346888", "#5886a5", "#7aa6c2", "#9dc6e0", "#c1e7ff"][::-1]

    fig = barhchart(words, scores, colors=colors, title="")
    return fig

# generate a barchart using plotly
def barhchart(labels: List[str],
                values: List[int],
                colors: List[str],
                title: str = "<b>Emotion Distribution</b>",
                font_color = "black",
                font_size = 14,
                width: int = 250,
                height: int = 250) -> go.Figure:
    """ Visualize a horizontal barchart

    Arguments:
        labels: Labels (y axis).
        values: Values for each label().
        y_label: Name y label
        x_label: Name x label
        title: Title of the plot.
        width: The width of each figure.
        height: The height of each figure.

    Returns:
        fig: A plotly figure
    """
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker = dict(color=colors)
        )
    )
    fig.update_layout(title=title)
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(showgrid=True)

    #change font color and size
    fig.update_traces(textfont=dict(color=font_color, size=font_size), selector=dict(type='bar'))

    return fig