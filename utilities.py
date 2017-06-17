
import numpy as np
nan = float('nan')
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit as sss, ShuffleSplit as ss, GridSearchCV



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


def big_loop(models_n_params, x, y, isClassification,
             test_size = 0.2, n_splits = 5, random_state=None, doesUpsample=True,
             scoring='accuracy',
             verbose=False):
    """
    runs through all model classes with their perspective hyper parameters
    :param models_n_params: [(model class, hyper parameters),...]
    :param isClassification: whether it's a classification or regression problem
    :type isClassification: bool
    :return: the best estimator, list of [(estimator, cv score),...]
    """

    res = []
    cv = cv_clf(x, y, test_size, n_splits, random_state, doesUpsample) \
        if isClassification \
        else cv_reg(x, test_size, n_splits, random_state)

    for clf_Klass, parameters in models_n_params:

        clf_search = GridSearchCV(clf_Klass(), parameters, scoring, cv=cv)
        clf_search.fit(x, y)

        print('--------', clf_Klass.__name__)

        print('best score:', clf_search.best_score_)
        print('best params:', clf_search.best_params_)

        if verbose:
            print('validation scores:', clf_search.cv_results_['mean_test_score'])
            print('training scores:', clf_search.cv_results_['mean_train_score'])

        res.append((clf_search.best_estimator_, clf_search.best_score_))

    winner = res[np.argmax([v[1] for v in res])][0]
    print('='*15)
    print('The winner is:', winner.__class__.__name__)

    return winner, res


if __name__ == '__main__':

    y = np.array([0,1,0,0,0,3,1,1,3])
    x = np.zeros(len(y))

    for t, v in cv_reg(x):
        print('---------')
        print('training inds:', t)
        print('valid inds:', v)

        # for t, v in cv_clf(x, y, test_size=3):
        #     print('---------')
        #     print('training inds:', t)
        #     print('valid inds:', v)