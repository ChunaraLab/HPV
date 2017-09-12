"""
ml.pipeline
~~~~~~~~~~~

Its important to instantiate the pipeline and use the grid parameters with gridsearch
to find some kind of optimal paramater for the classifier.

"""
__author__ = 'JasonLiu'

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from scipy.stats import randint


def get_pipeline(clf):
    """
    :param clf: classifier
    :return: Pipeline that contains tfidf with additional topic information
    """
    return Pipeline([
        ("features",
         FeatureUnion([
             ("text", TfidfVectorizer()),
             # ("topics", Pipeline([
             #     ("text", TfidfVectorizer()),
             #     ("lsi", TruncatedSVD(n_components=50))
             # ]))
         ])),
        ("clf", clf())
    ])

def get_text_parameters():
    """
    :return: basic set of grid parameters for features__text and featres_topics
    """
    return {
        'features__text__analyzer':
            ['char', 'word'],
        'features__text__lowercase':
            [False, True],
        'features__text__max_features':
            randint(70000, 100000),
        'features__text__min_df':
            randint(1, 5),
        'features__text__ngram_range': [
            (1, 1),
            (1, 3),
            (2, 5),
            (2, 8)],
        'features__text__norm':
            ['l2', 'l1'],
        'features__text__stop_words':
            [None, "english"],
        # 'features__topics__text__analyzer':
        #     ['char', 'word'],
        # 'features__topics__text__lowercase':
        #     [False,True],
        # 'features__topics__text__max_features':
        #     randint(70000, 100000),
        # 'features__topics__text__min_df':
        #     randint(1, 5),
        # 'features__topics__text__ngram_range': [
        #     (1, 1),
        #     (1, 3),
        #     (2, 5),
        #     (2, 8)],
        # 'features__topics__text__norm':
        #     ['l2', 'l1'],
    }
