# WHAT IT IS
Many believe that 

> most of the work of supervised (non-deep) Machine Learning lies in feature engineering, whereas the model-selection process is just running through all the models with a huge for-loop. 

So one glorious weekend, I decided to type out said loop!

# HOW IT WORKS
Runs through all `sklearn` models (both classification and regression), with **all possible hyperparameters**, and rank using cross-validation.

# MODELS
Runs **all the model** available on `sklearn` for supervised learning [here](http://scikit-learn.org/stable/supervised_learning.html). The categories are:

* Generalized Linear Models
* Kernel Ridge
* Support Vector Machines
* Nearest Neighbors
* Gaussian Processes
* Naive Bayes
* Trees
* Neural Networks
* Ensemble methods

Note: I skipped GradientTreeBoosting due to sub-par model performance, long run-time and constant convergence issues. Skipped AdaBoost because it keeps giving max_features errors. (Please ping me or feel free to contribute to the repo directly if you ever got AdaBoost to work at some point.)

# USAGE

### How to run
1. Feed in `X` (2-D `numpy.array`) and `y` (1-D `numpy.array`). (The code also has fake data generated for testing purposes.)
2. Use `run_classification` or `run_regression` where appropriate. 

### Knobs

* Evaluation criteria

  By default classification uses accuracy and regression uses negative MSE, given by the parameter of the `big_loop` function in `utilities.py`. It also accepts any `sklearn` scoring string.

* Scale

  Because it takes a long time to run through all models and hyperparameters at full-blown scale, there is a "small" and a full version of hyperparameters for almost every model. The "small" ones run much faster by evaluating only the most essential hyperparameters in smaller ranges than the full version. It's controlled by the `small` parameter of all of the `run_all` functions.

* Hyperparameters

  You can modify the search space of hyperparameters in `run_regression.py` and `run_classification.py`.

The output looks this:

| Model                       |  accuracy     |  Time/clf (s)|
|---------------------------- |:-------------:|:-------------:|
|SGDClassifier                |     0.967     |      0.001   |
|LogisticRegression           |     0.940      |      0.001   |
|Perceptron                   |     0.900       |      0.001   |
|PassiveAggressiveClassifier  |     0.967     |      0.001   |
|MLPClassifier                |     0.827     |      0.018   |
|KMeans                       |     0.580      |      0.010    |
|KNeighborsClassifier         |     0.960      |      0.000       |
|NearestCentroid              |     0.933     |      0.000       |
|RadiusNeighborsClassifier    |     0.927     |      0.000       |
|SVC                          |     0.960      |      0.000       |
|NuSVC                        |     0.980      |      0.001   |
|LinearSVC                    |     0.940      |      0.005   |
|RandomForestClassifier       |     0.980      |      0.015   |
|DecisionTreeClassifier       |     0.960      |      0.000       |
|ExtraTreesClassifier         |     0.993     |      0.002   |

*The winner is: ExtraTreesClassifier with score 0.993.*

# TO-DO'S

Feel free to contribute by hashing out the following:

* Wrap an emsemble (bagging/boosting) model on top of the best models.
* multi-target classification (i.e. `y` having multiple columns)

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

Oh boy that was a lot of typing in the past 24 hours! Hopefully it saves you (and myself) some typing in the future. I'm gonna grab some lunch, sip a cold drink and enjoy the California summer heat. :)
Check out more of my pet projects on [planetj.io](planetj.io).
