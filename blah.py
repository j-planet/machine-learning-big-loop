"""
model selection
"""

import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
nan = float('nan')

from utilities import x_y_cv
from read_data import read_data


def model_selection(x, y, cv, verbose=True):

    res = []

    for clf_Klass, parameters in [
        (linear_model.SGDClassifier,
         {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
          'alpha': [0.0001, 0.001, 0.1],
          'penalty': ['l1', 'l2', 'none']}),

        (RandomForestClassifier,
         {'n_estimators': [2, 3, 5, 10, 25],
          'max_features': [3, 5, 7, 'auto', 'log2'],
          'min_samples_leaf': [2, 3]}),

        (GradientBoostingClassifier,
         {'n_estimators': [3, 5, 10, 25],
          'max_depth': [2, 3, 4],
          'max_features': [3, 5, 7, 'auto', 'log2', None]}),

        (KMeans,
         {'n_clusters': [2],
          'algorithm': ['auto', 'full', 'elkan'],
          'init': ['k-means++', 'random']}),

        (KNeighborsClassifier,
         {'n_neighbors': [5, 7, 10, 15, 20],
          'weights': ['uniform', 'distance'],
          'algorithm': ['ball_tree', 'kd_tree', 'brute'],
          'leaf_size': [2, 3, 5, 10],
          'p': [1, 2]})
    ]:

        clf_search = GridSearchCV(clf_Klass(), parameters, cv=cv, scoring='recall')
        clf_search.fit(x, y)

        print('--------', clf_Klass.__name__)
        valid_scores = clf_search.cv_results_['mean_test_score']

        print('best score:', max(valid_scores))
        print('best params:', clf_search.cv_results_['params'][np.argmax(valid_scores)])

        if verbose:
            print('validation scores:', valid_scores)
            print('training scores:', clf_search.cv_results_['mean_train_score'])

        res.append((clf_search.best_estimator_, clf_search.best_score_))

    return res[np.argmax([v[1] for v in res])][0], res

if __name__ == '__main__':

    raw, agg, numdupes = read_data()
    x, y, x_columns, cv = x_y_cv(agg, 1/7)

    clf, res = model_selection(x['train_valid'], y['train_valid'], cv, verbose=False)
