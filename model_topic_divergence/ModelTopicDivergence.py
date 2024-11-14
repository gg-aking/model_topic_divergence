from collections import defaultdict, Counter
import pandas as pd
import numpy as np

from model_topic_divergence.VertexTopicNamer import VertexTopicNamer
from model_topic_divergence.topic_models import BertTopicModel, TopicModelSelector

class ModelTopicDivergence:

  def __init__(self, df : pd.DataFrame,
                     label_1_col : str,
                     label_2_col : str,
                     text_col : str = 'text',
                     text_lang : str = 'english',
                     automatic_llm_label_names : bool = True,
                     vertex_kwargs = {},
                     ):
    
    self.df = df.copy()
    self.label_1_col = label_1_col
    self.label_2_col = label_2_col
    self.text_col = text_col
    self.text_lang = text_lang
    self.automatic_llm_label_names = automatic_llm_label_names
    self.vertex_kwargs = vertex_kwargs

    self.vtn = VertexTopicNamer(**vertex_kwargs)
    self.df = df.dropna(subset = [self.text_col]).reset_index(drop = True).copy()
    # filter out rows with empty or very short text
    self.df = self.df[self.df[self.text_col].str.len() >= 10].reset_index(drop = True).copy()

    # fit multiple cluster models, pick the one with the lowest within-topic entropy
    self.tms = TopicModelSelector(self.df, self.text_col, 
                                  self.label_1_col, self.label_2_col)
    self.btm, self.n_clusters = self.tms.find_best_model()

    self.topics, self.probs = self.btm.topics, self.btm.probs
    self.topic_model = self.btm.topic_model
    
    self.topic_counter = Counter(self.topics)
    self.df['topic'] = self.topics
    self.grouped_df = self.df[['topic', 
                               self.label_1_col, self.label_2_col]].groupby('topic').agg(lambda l: Counter(l)[True]).reset_index()
    self.grouped_df['topic_count'] = self.grouped_df['topic'].apply(self.topic_counter.get)
    self.grouped_df['ratio'] = self.grouped_df.apply(lambda row : row[self.label_1_col] / max(1, row[self.label_2_col]), 
                                                    axis = 1)
    self.grouped_df = self.grouped_df.sort_values('ratio').reset_index(drop = True)
    self.topic2name = {}
    
    # use LLM to generate label names. Don't always do to avoid unneeded LLM calls
    if self.automatic_llm_label_names:
      self.get_label_names_for_top_n_topics()

  def get_example_for_topic(self, topic_id : int, 
                            n_examples : int = 1,
                            alternate_text_col : str = None,
                            shortest_first : bool = True):
    """
      Get a sample of `n_examples` examples from the given topic. 
      If `alternate_text_cold` is provided, return examples from that column instead.
    """
    topic_sub_df = self.df[self.df['topic'] == topic_id].drop_duplicates(subset = [self.text_col]).reset_index(drop = True).copy()
    if shortest_first:
      topic_sub_df = topic_sub_df.sort_values(self.text_col, key=lambda s: s.str.len()).reset_index(drop = True)
      topic_sub_df = topic_sub_df.head(len(topic_sub_df) // 2)
    
    n_examples = min(n_examples, len(topic_sub_df))
    topic_sub_df = topic_sub_df.sample(n = n_examples).reset_index(drop = True)
    if alternate_text_col is not None:
      return topic_sub_df[alternate_text_col].tolist()
    else:
      return topic_sub_df[self.text_col].tolist()

  def get_label_names_for_top_n_topics(self, n : int = 8, 
                                       skip_based_on_ratio : bool = True):
    """
      Get label names for the top `n` topics based on the ratio of `label_1_col` to `label_2_col`.
      Ratio = the proportion of examples in that topic that are true for one set of labels and not the other.
    """
    for i, row in self.grouped_df.head(n).iterrows():
      topic_id = int(row['topic'])
      if skip_based_on_ratio and row['ratio'] >= 1:
        continue
      if topic_id not in self.topic2name:
        topic_kws = self.topic_model.get_topic(topic_id)
        topic_kws = [t[0] for t in topic_kws]
        self.topic2name[topic_id] = self.vtn.gen_topic_label_name(topic_kws)
    
    for i, row in self.grouped_df.tail(n).iterrows():
      topic_id = int(row['topic'])
      if skip_based_on_ratio and row['ratio'] < 1:
        continue
      if topic_id not in self.topic2name:
        topic_kws = self.topic_model.get_topic(topic_id)
        topic_kws = [t[0] for t in topic_kws]
        self.topic2name[topic_id] = self.vtn.gen_topic_label_name(topic_kws)
    
    self.grouped_df['topic_name'] = self.grouped_df['topic'].apply(lambda t : self.topic2name.get(int(t), ''))