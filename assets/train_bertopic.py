from typing import List, Iterable
from bertopic._utils import check_is_fitted
import pandas as pd
import math
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from bertopic import BERTopic
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from bertopic.representation import MaximalMarginalRelevance
import numpy as np
from transformers.pipelines import pipeline


# modified Bertopic class to save all documents (up to 10000) instead of 3 representative ones
class BERTopicAdjusted(BERTopic):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    def _save_representative_docs(self, documents: pd.DataFrame):
        """ Save the 10000 most representative docs per topic (instead of 3)
        Arguments:
            documents: Dataframe with documents and their corresponding IDs
        Updates:
            self.representative_docs_: Populate each topic with up to 10000 representative docs
        """
        repr_docs, _, _ = self._extract_representative_docs(self.c_tf_idf_,
                                                            documents,
                                                            self.topic_representations_,
                                                            nr_samples=500,
                                                            nr_repr_docs=10000)
        self.representative_docs_ = repr_docs

    # merge topics without needing to pass the docs as argument
    def merge_topics_no_docs(self,
                             topics_to_merge: List[int]) -> None:
        """
        Arguments:
            topics_to_merge: A list of topics to merge. For example:
                                [1, 2, 3] will merge topics 1, 2 and 3

        Examples:

        If you want to merge topics 1, 2, and 3:

        ```python
        topics_to_merge = [1, 2, 3]
        topic_model.merge_topics_no_docs(topics_to_merge)
        ```
        """
        check_is_fitted(self)

        docs = []
        topics = []
        for topic_number, representative_docs in self.representative_docs_.items():
            for d in representative_docs:
                docs.append(d)
                topics.append(topic_number)

        documents = pd.DataFrame({"Document": docs, "Topic": topics})

        mapping = {topic: topic for topic in set(self.topics_)}
        if isinstance(topics_to_merge[0], int):
            for topic in sorted(topics_to_merge):
                mapping[topic] = topics_to_merge[0]
        elif isinstance(topics_to_merge[0], Iterable):
            for topic_group in sorted(topics_to_merge):
                for topic in topic_group:
                    mapping[topic] = topic_group[0]
        else:
            raise ValueError("Make sure that `topics_to_merge` is either"
                             "a list of topics or a list of list of topics.")

        documents.Topic = documents.Topic.map(mapping)
        self.topic_mapper_.add_mappings(mapping)
        documents = self._sort_mappings_by_frequency(documents)
        self._extract_topics(documents)
        self._update_topic_size(documents)
        self._save_representative_docs(documents)
        self.probabilities_ = self._map_probabilities(self.probabilities_)


# get list of stopwords and choose transformer for the embeddings depending on the language
def select_stopwords_and_embedding_model(language: str, transformer_model: str):
    stop_words = None
    embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"  # default multilingual model

    if language == "english":
        # TO DO: check effect min_df=10
        stop_words = "english"
        embedding_model = "all-MiniLM-L6-v2"  # embedding model all-MiniLM-L6-v2 which performs better for English texts
    elif language == "dutch":
        stop_words = stopwords.words('dutch')
        # DUTCH BERT MODELS -> worse_performance
        if transformer_model == "robBERT":
            embedding_model = pipeline("feature-extraction",
                                       model="pdelobelle/robbert-v2-dutch-base")  # robBERT model for dutch language
        elif transformer_model == "BERTje":
            embedding_model = pipeline("feature-extraction", model="GroNLP/bert-base-dutch-cased")
    elif language == "multilingual":
        # Assumption: languages used are Dutch, English or German
        stop_words = stopwords.words('dutch') + stopwords.words('english') + stopwords.words('german')
    # another language
    else:
        # if language supported by nltk stopwords, get those stopwords
        if language in stopwords.fileids():
            stop_words = stopwords.words(language)
    return stop_words, embedding_model


def train_model(docs,
                language: str = "multilingual",
                transformer_model=None,
                n_gram=(1, 3),
                min_df: int = 10,
                get_metrics: bool = True,
                save_model: bool = True,
                model_name="topic_model",
                top_k_words_diversity: int = 10) -> BERTopicAdjusted:
    # Set random_state in umap to avoid stochastic behavior
    umap_model = UMAP(n_neighbors=15, n_components=5,
                      min_dist=0.0, metric='cosine', random_state=251)
    stop_words = None
    embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"  # default multilingual model

    stop_words, embedding_model = select_stopwords_and_embedding_model(language=language,
                                                                       transformer_model=transformer_model)

    vectorizer_model = CountVectorizer(ngram_range=n_gram,
                                       stop_words=stop_words,
                                       min_df=min_df)  # n_gram -> decide how many tokens each entity is in a topic representation

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    topic_model = BERTopicAdjusted(embedding_model=embedding_model,
                                   umap_model=umap_model,
                                   vectorizer_model=vectorizer_model,
                                   ctfidf_model=ctfidf_model,
                                   representation_model=representation_model,
                                   language=language, verbose=False,
                                   low_memory=True)
    topics, probs = topic_model.fit_transform(docs)

    if save_model:
        model_path = "models/" + model_name
        topic_model.save(path=model_path)

    if get_metrics:
        topic_words = [[words for words, _ in topic_model.get_topic(topic)]
                       for topic in range(len(set(topics)) - 1)]
        model_output = {'topics': []}
        for key in topic_model.get_topics():
            words = [word + "  " for word, _ in topic_model.get_topic(key)][:top_k_words_diversity][::-1]
            model_output['topics'].append(words)
        calculate_metrics(vectorizer=vectorizer_model,
                          topic_words=topic_words,
                          docs=docs, topics=topics,
                          probs=probs,
                          model_output=model_output,
                          top_k_words_diversity=top_k_words_diversity)

    return topic_model


def calculate_perplexity(probs: List[float]) -> float:
    print("--------------------------------------")
    print("PERPLEXITY")
    log_perplexity = -1 * np.mean(np.log(np.sum(probs)))
    perplexity = np.exp2(log_perplexity)
    print("res: ", perplexity)
    print("--------------------------------------")
    return perplexity


def calculate_umass(topics,
                    texts,
                    corpus,
                    dictionary: corpora.Dictionary) -> float:
    c_umass_model = CoherenceModel(topics=topics,
                                   texts=texts,
                                   corpus=corpus,
                                   dictionary=dictionary,
                                   coherence='u_mass')
    # calculate coherence umass directly
    umass = c_umass_model.get_coherence()

    # get umass per topic
    confirmed_measures = c_umass_model.get_coherence_per_topic()
    # delete nans
    confirmed_measures_no_nans = [item for item in confirmed_measures if not (math.isnan(item)) == True]
    # check if any scores were deleted (nans)
    print(f"deleted {len(confirmed_measures) - len(confirmed_measures_no_nans)} elements")
    # aggregate score
    umass_ignoring_nans = c_umass_model.aggregate_measures(confirmed_measures_no_nans)

    print("res coherence umass: ", umass, umass_ignoring_nans)  # None they are either equal or umass is nan
    return umass_ignoring_nans


def calculate_c_v(topics,
                  texts,
                  dictionary: corpora.Dictionary) -> float:
    c_v_model = CoherenceModel(topics=topics,
                               texts=texts,
                               dictionary=dictionary,
                               coherence='c_v')
    c_v = c_v_model.get_coherence()
    print("res coherence c_v: ", c_v)
    return c_v


def calculate_c_uci(topics,
                    texts,
                    corpus,
                    dictionary: corpora.Dictionary) -> float:
    c_uci_model = CoherenceModel(topics=topics,
                                 texts=texts,
                                 corpus=corpus,
                                 dictionary=dictionary,
                                 coherence='c_uci')

    c_uci = c_uci_model.get_coherence()
    confirmed_measures = c_uci_model.get_coherence_per_topic()
    confirmed_measures_no_nans = [item for item in confirmed_measures if not (math.isnan(item)) == True]
    print(f"deleted {len(confirmed_measures) - len(confirmed_measures_no_nans)} elements")
    c_uci_ignoring_nans = c_uci_model.aggregate_measures(confirmed_measures_no_nans)

    print("res coherence c_uci: ", c_uci, c_uci_ignoring_nans)
    return c_uci_ignoring_nans


def calculate_c_npmi(topics,
                     texts,
                     dictionary: corpora.Dictionary) -> float:
    c_npmi_model = CoherenceModel(topics=topics,
                                  texts=texts,
                                  dictionary=dictionary,
                                  coherence='c_npmi')

    c_npmi = c_npmi_model.get_coherence()
    confirmed_measures = c_npmi_model.get_coherence_per_topic()
    print("len", len(confirmed_measures))
    confirmed_measures_no_nans = [item for item in confirmed_measures if not (math.isnan(item)) == True]
    print(f"deleted {len(confirmed_measures) - len(confirmed_measures_no_nans)} elements")
    c_npmi_ignoring_nans = c_npmi_model.aggregate_measures(confirmed_measures_no_nans)
    print("res coherence c_npmi: ", c_npmi, c_npmi_ignoring_nans)


# source: https://www.theanalyticslab.nl/topic-modeling-with-bertopic/
def calculate_coherence(vectorizer,
                        topic_words,
                        docs,
                        topics):
    print("--------------------------------------")
    print("COHERENCE")

    documents = pd.DataFrame({"Document": docs,
                              "ID": range(len(docs)),
                              "Topic": topics})

    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = BERTopic()._preprocess_text(documents=documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    # words = vectorizer.get_feature_names_out()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    calculate_umass(topics=topic_words, texts=tokens, corpus=corpus, dictionary=dictionary)
    calculate_c_v(topics=topic_words, texts=tokens, dictionary=dictionary)
    calculate_c_uci(topics=topic_words, texts=tokens, corpus=corpus, dictionary=dictionary)
    calculate_c_npmi(topics=topic_words, texts=tokens, dictionary=dictionary)

    print("--------------------------------------")


def calculate_topic_diversity(model_output, n_words: int = 10):
    print("--------------------------------------")
    print("TOPIC DIVERSITY")
    metric = TopicDiversity(topk=n_words)

    diversity_score = metric.score(model_output)
    print("res topic diversity", diversity_score)
    print("--------------------------------------")
    return diversity_score


def calculate_metrics(vectorizer, topic_words, docs: List[str], topics, probs, model_output, top_k_words_diversity):
    calculate_perplexity(probs=probs)
    calculate_coherence(vectorizer=vectorizer, topic_words=topic_words, docs=docs, topics=topics)
    calculate_topic_diversity(model_output=model_output, n_words=top_k_words_diversity)
