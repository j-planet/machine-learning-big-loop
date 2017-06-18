from pprint import pprint
import numpy as np
nan = float('nan')
from collections import Counter
from multiprocessing import cpu_count
from time import time
from tabulate import tabulate

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit as sss, ShuffleSplit as ss, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


TREE_N_ENSEMBLE_MODELS = [RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor]

def upsample_indices_clf(inds, y):
    """
    make all classes have the same number of samples. for classification only.
    :type inds: numpy array
    :type y: numpy array
    :return: a numpy array of indices
    """

    assert len(inds) == len(y)

    countByClass = dict(Counter(y))
    maxCount = max(countByClass.values())

    extras = []

    for klass, count in countByClass.items():
        if maxCount == count: continue

        ratio = int(maxCount / count)
        cur_inds = inds[y == klass]

        extras.append(np.concatenate(
            (np.repeat(cur_inds, ratio - 1),
             np.random.choice(cur_inds, maxCount - ratio * count, replace=False)
             ))
        )

        print('upsampling class %d, %d times' % (klass, ratio-1))

    return np.concatenate((inds, *extras))


def cv_clf(x, y,
           test_size = 0.2, n_splits = 5, random_state=None,
           doesUpsample = True):
    """
    an iterator of cross-validation groups with upsampling
    :param x:
    :param y:
    :param test_size:
    :param n_splits:
    :return:
    """

    sss_obj = sss(n_splits, test_size, random_state=random_state).split(x, y)

    # no upsampling needed
    if not doesUpsample:
        return sss_obj

    # with upsampling
    for train_inds, valid_inds in sss_obj:
        yield (upsample_indices_clf(train_inds, y[train_inds]), valid_inds)


def cv_reg(x, test_size = 0.2, n_splits = 5, random_state=None):
    return ss(n_splits, test_size, random_state=random_state).split(x)

def timeit(klass, params, x, y):
    """
    time in seconds
    """

    start = time()
    clf = klass(**params)
    clf.fit(x, y)

    return time() - start

def big_loop(models_n_params, x, y, isClassification,
             test_size = 0.2, n_splits = 5, random_state=None, doesUpsample=True,
             scoring=None,
             verbose=False, n_jobs = cpu_count()-1):
    """
    runs through all model classes with their perspective hyper parameters
    :param models_n_params: [(model class, hyper parameters),...]
    :param isClassification: whether it's a classification or regression problem
    :type isClassification: bool
    :param scoring: by default 'accuracy' for classification; 'neg_mean_squared_error' for regression
    :return: the best estimator, list of [(estimator, cv score),...]
    """

    def cv_():
        return cv_clf(x, y, test_size, n_splits, random_state, doesUpsample) \
            if isClassification \
            else cv_reg(x, test_size, n_splits, random_state)

    res = []
    num_features = x.shape[1]
    scoring = scoring or ('accuracy' if isClassification else 'neg_mean_squared_error')
    print('Scoring criteria:', scoring)

    for i, (clf_Klass, parameters) in enumerate(models_n_params):
        try:
            print('-'*15, 'model %d/%d' % (i+1, len(models_n_params)), '-'*15)
            print(clf_Klass.__name__)

            if clf_Klass == KMeans:
                parameters['n_clusters'] = [len(np.unique(y))]
            elif clf_Klass in TREE_N_ENSEMBLE_MODELS:
                parameters['max_features'] = [v for v in parameters['max_features']
                                              if v is None or type(v)==str or v<=num_features]

            clf_search = GridSearchCV(clf_Klass(), parameters, scoring, cv=cv_(), n_jobs=n_jobs)
            clf_search.fit(x, y)

            timespent = timeit(clf_Klass, clf_search.best_params_, x, y)
            print('best score:', clf_search.best_score_, 'time/clf: %0.3f seconds' % timespent)
            print('best params:')
            pprint(clf_search.best_params_)

            if verbose:
                print('validation scores:', clf_search.cv_results_['mean_test_score'])
                print('training scores:', clf_search.cv_results_['mean_train_score'])

            res.append((clf_search.best_estimator_, clf_search.best_score_, timespent))

        except Exception as e:
            print('ERROR OCCURRED')
            if verbose: print(e)
            res.append((clf_Klass(), -np.inf, np.inf))


    print('='*60)
    print(tabulate([[m.__class__.__name__, '%.3f'%s, '%.3f'%t] for m, s, t in res], headers=['Model', scoring, 'Time/clf (s)']))
    winner_ind = np.argmax([v[1] for v in res])
    winner = res[winner_ind][0]
    print('='*60)
    print('The winner is: %s with score %0.3f.' % (winner.__class__.__name__, res[winner_ind][1]))

    return winner, res


if __name__ == '__main__':

    y = np.array([0,1,0,0,0,3,1,1,3])
    x = np.zeros(len(y))

    for t, v in cv_reg(x):
        print('---------')
        print('training inds:', t)
        print('valid inds:', v)

    for t, v in cv_clf(x, y, test_size=3):
        print('---------')
        print('training inds:', t)
        print('valid inds:', v)