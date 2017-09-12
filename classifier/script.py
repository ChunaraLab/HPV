import pandas as pd

from ml.pipeline import get_pipeline, get_text_parameters

from scipy.stats import uniform

from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()

    df = pd.DataFrame.from_csv("./template_marked_tweets.csv")
    df = df.reset_index()

    X= df.t 

    y=df.newtag.astype(int)
    y.value_counts()

    #print y
    pipe = get_pipeline(SVC)

    text_params = get_text_parameters()

    svc_params = {
        'clf__C': uniform(10 ** -4, 10 ** 4),
        'clf__cache_size': [500],
        #'clf__class_weight': ['auto', None],
        'clf__coef0': uniform(0, 1),
        'clf__degree': [1, 2, 3,4,5,6,7,8],
        #'clf__gamma': ['auto'],
        'clf__kernel': ['poly','rbf'],
        'clf__probability': [False],
        'clf__tol': uniform(0.0001, 0.001)
    }

    svc_params.update(text_params)

    clf = RandomizedSearchCV(pipe, svc_params, n_iter=30, scoring="accuracy", n_jobs=4, verbose=1)

    clf.fit(X,y)
    
    #z=clf.predict(X)

    params_scores = []

    params_scores.append((clf.best_score_, clf.best_params_))
    
    print clf.best_score_
    print clf.best_params_
    
    pipe2 = get_pipeline(SVC)

    pipe2.set_params(**clf.best_params_)
    
    
    pipe2.fit(X,y)

    dd = pd.DataFrame.from_csv("./template_unmarked_tweets.csv").reset_index()
    #print 11
    y_pred_new = pipe2.predict(dd.t)
    #y_pred_new = pipe2.predict(X)
    
    for row in y_pred_new:
        with open("predicted_tag.csv", "a") as myfile:
            myfile.write('"{0}"\n'.format(row))
            
    
