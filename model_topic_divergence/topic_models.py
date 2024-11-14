from collections import defaultdict, Counter
from typing import List, Optional

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance

def calc_entropy(l: List[int]) -> float:
    """
    Calculate the entropy of a distribution.

    This function computes the entropy `H` of a given list of counts, `l`, 
    representing the frequency distribution of a variable. The entropy is 
    calculated using the formula:

        H = -sum([p * log2(p) for p in P])

    where `p` represents the probability of each count in `l`.

    Args:
        l (List[int]): A list of integer counts representing a distribution.

    Returns:
        float: The entropy value of the distribution, a measure of uncertainty 
               or randomness in the distribution.
    """
    P = [l_ / sum(l) for l_ in l]
    H = np.sum([p * -np.log2(p) for p in P])
    return H
  
class BertTopicModel:
  def __init__(self,
               representation_model_kwargs = {'diversity' : 0.2},
               cluster_model_kwargs = {'n_clusters' : 32},
               topic_model_lang = 'en',
               topic_model_kwargs = {'top_n_words' : 30, 'min_topic_size' : 5, 
                                     'calculate_probabilities' : True, 'verbose' : False},
               ):
               
    self.representation_model_kwargs = representation_model_kwargs
    self.cluster_model_kwargs = cluster_model_kwargs
    # hard coding stuff for Japanese b/c that was the original use of this project.
    # to-do: make more language agnostic/able to do other languages
    self.topic_model_lang = 'japanese' if topic_model_lang.startswith('ja') else 'english'
    self.topic_model_kwargs = topic_model_kwargs


    # this helps make the keywords per label more diverse, which helps create more unique and descriptive
    # label names
    self.representation_model = MaximalMarginalRelevance(**self.representation_model_kwargs)
    # use KMeans over default DBSCAN. DBSCAN tends to create unequal topic clusters with a lot of -1 outliers which
    # is less useful than having a fixed number of more or less equally sized topics
    self.cluster_model = KMeans(**self.cluster_model_kwargs)
    self.topic_model = BERTopic(language=self.topic_model_lang,
                                representation_model=self.representation_model,
                                hdbscan_model=self.cluster_model,
                                vectorizer_model=CountVectorizer(stop_words='english') if self.topic_model_lang == 'english' else None,
                                **self.topic_model_kwargs)
    
  def fit_transform(self, texts : List[str]):
    self.topics, self.probs = self.topic_model.fit_transform(texts)
    return self.topics, self.probs


class TopicModelSelector:

  def __init__(self,
               df : pd.DataFrame,
               text_col : str,
               label_1_col : str,
               label_2_col : str,
               paired_lagrange : float = .75,
    ):
    """
      LaGrange multiplier:
       - .5: topics will be relatively balanced between the two labels
       - close to 1: topics will be purer in terms of when the two models agree or disagree within a topic
       - close to 0: topics will be organized so that topics agree more within each individual set of labels but not across models
    """
    self.df = df
    self.text_col = text_col
    self.label_1_col = label_1_col
    self.label_2_col = label_2_col
    self.paired_lagrange = paired_lagrange

  def find_best_model(self):
    """
    Identify the optimal clustering model and number of clusters based on topic entropy.

    This function iterates over a range of cluster counts, fitting a `BertTopicModel` 
    for each specified `n_clusters` value. For each model, it calculates entropy scores 
    using three methods: paired labels, `label_1`, and `label_2`. The function then 
    combines these entropy scores, weighted by `self.paired_lagrange`, to compute 
    a composite entropy measure (`H`). The model with the lowest entropy value is selected.

    Returns:
        Tuple[BertTopicModel, int]: A tuple containing the best `BertTopicModel` instance 
                                    and the optimal number of clusters (`n_clusters`) 
                                    that minimizes the combined entropy score.
    """
    list_of_H = []
    
    ###
    # to:do - make this smarter
    search_range = range(8, max(8, min(32, len(self.df) // 128)) + 1, 4)
    ###

    for n_clusters in search_range:
      print('\t', n_clusters)
      btm = BertTopicModel(cluster_model_kwargs = {'n_clusters' : n_clusters})
      topics, probs = btm.fit_transform(self.df[self.text_col])
      # calculate the entropy for agreement between the two labels per topic
      # if each topic is made up of only identical kinds of pairs of labels, the entropy is 0
      # if there is a lot of disagreement, the entropy will be higher
      paired_H = self.calculate_topic_H(topics, 
                                self.df[self.label_1_col], self.df[self.label_2_col])
      # calculate entropy for each individual set of labels for each topic
      # entropy will be close to 0 if each topic is made up of only one kind of pair of labels, i.e., True or False
      label_1_H = self.calculate_topic_H(topics, self.df[self.label_1_col])
      label_2_H = self.calculate_topic_H(topics, self.df[self.label_2_col])
      
      H = (self.paired_lagrange * paired_H) + ((1 - self.paired_lagrange) * np.mean([label_1_H, label_2_H]))
      print('\t\t', paired_H, label_1_H, label_2_H, H)
      list_of_H.append((H, n_clusters, btm))

    best_result = sorted(list_of_H)[0]
    H, btm, n_clusters = best_result
    print(H, n_clusters)
    return btm, n_clusters
  
  def calculate_topic_H(self, topics : List, 
                        labels_1 : List[bool], 
                        labels_2 : Optional[List[bool]] = None):
    """
    Calculate the weighted mean entropy across topics based on label distributions.

    This function computes the entropy of each topic in the `topics` list by considering 
    the frequency distribution of label combinations. If `labels_2` is provided, the 
    entropy calculation uses pairs of labels `(label_1, label_2)`. If `labels_2` is not 
    provided, it uses only `label_1`.

    Args:
        topics (List): A list of topics corresponding to data points.
        labels_1 (List[bool]): A list of binary labels (first set of labels) for each topic.
        labels_2 (Optional[List[bool]]): An optional second list of binary labels. If provided, 
                                         entropy is calculated based on pairs `(label_1, label_2)`. 

    Returns:
        float: The weighted mean entropy across all topics, where the weight of each topic's 
               entropy is proportional to its frequency in the `topics` list.
    """
    topic_counts = Counter(topics)
    topic2label_difs = defaultdict(Counter)
    if labels_2 is not None:
      for i, (topic, label_1, label_2) in enumerate(zip(topics, labels_1, labels_2)):
        topic2label_difs[topic][(label_1, label_2)] += 1 
    else:
      for i, (topic, label_1) in enumerate(zip(topics, labels_1)):
        topic2label_difs[topic][label_1] += 1 

    topic2entropy = dict((topic, calc_entropy(l.values())) for topic, l in topic2label_difs.items())
    topic2p = dict((topic, c / sum(topic_counts.values())) for topic, c in topic_counts.items())
    weighted_H = [topic2entropy[topic] * topic2p[topic] for topic in topic2entropy.keys()]
    mean_H = np.mean(weighted_H)
    return mean_H