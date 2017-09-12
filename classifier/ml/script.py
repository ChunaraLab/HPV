import pandas as pd

from ml.pipeline import get_pipeline, get_text_parameters

from scipy.stats import uniform

from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV


df = pd.DataFrame.from_csv("./marked_tweets.csv")
df = df.reset_index()

X,y = df.t, df.newtag

pipe = get_pipeline(SVC)

text_params = get_text_parameters()

svc_params = {
    'clf__C': uniform(10 ** -4, 10 ** 4),
    'clf__cache_size': [500],
    'clf__class_weight': ['auto', None],
    'clf__coef0': uniform(0, 1),
    'clf__degree': [1, 2, 3],
    'clf__gamma': ['auto'],
    'clf__kernel': ['poly', 'rbf'],
    'clf__probability': [False],
    'clf__tol': uniform(0.0001, 0.001)
}

svc_params.update(text_params)

clf = RandomizedSearchCV(pipe, svc_params, n_iter=30, scoring="f1_weighted", n_jobs=1, verbose=1)

clf.fit(X, y)

#params_scores = []

#params_scores.append((clf.best_score_, clf.best_params_))


#params_scores

