# source: https://towardsdatascience.com/topic-modelling-in-python-with-spacy-and-gensim-dc8f7748bdbf
import pandas as pd
import pyLDAvis.gensim_models
import spacy
import re
from tabulate import tabulate
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.test.utils import datapath
from gensim.models import CoherenceModel
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from typing import List
import numpy as np

NUM_TOPICS = [20, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# writes latex table with performance metrics for model trained with each number of topics
def main():
    list_perplexity, list_c_v, list_umass, list_c_uci, list_c_npmi, list_topic_diversity = [], [], [], [], [], []

    for n in NUM_TOPICS:
        print(f"number of topics is {n}")
        perplexity, c_v, umass, c_uci, c_npmi, topic_diversity = train_model(number_of_topics=n)

        list_perplexity.append(perplexity)
        list_c_v.append(c_v)
        list_umass.append(umass)
        list_c_uci.append(c_uci)
        list_c_npmi.append(c_npmi)
        list_topic_diversity.append(topic_diversity)

    df = pd.DataFrame({
        'Number of Topics': NUM_TOPICS,
        'Perplexity': [round(elem, 3) for elem in list_perplexity],
        'Umass': [round(elem, 3) for elem in list_umass],
        'C_v': [round(elem, 3) for elem in list_c_v],
        'C_uci': [round(elem, 3) for elem in list_c_uci],
        'C_npmi': [round(elem, 3) for elem in list_c_npmi],
        'Topic Diversity': [round(elem, 3) for elem in list_topic_diversity]
    })
    df.to_csv('gensim_results.csv', index=False)
    table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)

    with open('../gensim_results.tex', 'w') as f:
        f.write(table)

def train_model(number_of_topics: int, save_model: bool=False, generate_fig: bool=False) -> List[float]:
   data = pd.read_csv("../data/data_anon.csv")
   docs = data["LastMessage"]

   nlp = spacy.load("nl_core_news_sm")

   # tags to remove from the text
   removal = ['PRON', 'CCONJ', 'CONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM']

   # getting the tokens to analyze (all tokens that are alphabetical, not stop words and which tags
   # are not in the list of tags to remove)

   # texts will contain all tokens (to calculate coherence c_v)
   tokens = []
   texts = []

   for summary in nlp.pipe(docs):
      proj_tok = [token.lemma_.lower() for token in summary if
                  token.pos_ not in removal and not token.is_stop and token.is_alpha]
      all_tok = [token.lemma_.lower() for token in summary]
      tokens.append(proj_tok)
      texts.append(all_tok)

   data['tokens'] = tokens

   dictionary = Dictionary(data['tokens'])

   # filter out low-frequency and high-frequency tokens, also limit the vocabulary to a max of 1000 words
   dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)

   corpus = [dictionary.doc2bow(doc) for doc in data['tokens']]

   random_state = np.random.RandomState(1234)

   lda_model = LdaMulticore(corpus=corpus,
                            id2word=dictionary,
                            chunksize=2000,
                            passes=10,
                            iterations=400,
                            num_topics=number_of_topics,
                            workers=3,
                            random_state=random_state
                            )

   if save_model:
      path = f"lda_model_gensim_{number_of_topics}topics"
      model_file = datapath(path)
      lda_model.save(model_file)

   if generate_fig:
      lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
      fig = pyLDAvis.display(lda_display)
      pyLDAvis.save_html(lda_display, f"lda_model_gensim_{number_of_topics}topics.html")


   # Evaluation

   print("--------------------------------------")
   print("PERPLEXITY")

   log_perplexity = lda_model.log_perplexity(corpus)
   perplexity = np.exp2(log_perplexity)
   print("res: ", perplexity)

   print("--------------------------------------")
   print("COHERENCE")

   c_v_model = CoherenceModel(corpus=corpus,
                                    model=lda_model,
                                    texts=texts,
                                    dictionary=dictionary,
                                    coherence='c_v')
   c_v = c_v_model.get_coherence()
   print("res coherence c_v: ", c_v)

   model_umass = CoherenceModel(model=lda_model,
                                          corpus=corpus,
                                          coherence='u_mass')
   umass = model_umass.get_coherence()
   print("res coherence umass: ", umass)

   c_uci_model = CoherenceModel(corpus=corpus,
                                    model=lda_model,
                                    texts=texts,
                                    dictionary=dictionary,
                                    coherence='c_uci')

   c_uci = c_uci_model.get_coherence()
   print("res coherence c_uci: ", c_uci)

   c_npmi_model = CoherenceModel(corpus=corpus,
                                    model=lda_model,
                                    texts=texts,
                                    dictionary=dictionary,
                                    coherence='c_npmi')

   c_npmi = c_npmi_model.get_coherence()

   print("res coherence c_npmi: ", c_npmi)

   print("--------------------------------------")
   print("TOPIC DIVERSITY")

   n_words = 10
   metric = TopicDiversity(topk=n_words)

   output = lda_model.show_topics(num_topics=number_of_topics, num_words=n_words)

   top_words = []
   for topic, words in output:
      top_words.append(re.sub('[^A-Za-z ]+', '', words).split())

   model_output = {'topics': top_words}

   topic_diversity = metric.score(model_output)

   print("res topic diversity", topic_diversity)
   print("--------------------------------------")

   return perplexity, c_v, umass, c_uci, c_npmi, topic_diversity

if __name__ == '__main__':
   main()
