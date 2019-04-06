from functools import partial
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, Trials
from sklearn.model_selection import cross_validate, learning_curve


def objective(clf, features, truth, int_variables, params):
    start = timer()
    for variable in int_variables:
        params[variable] = int(params[variable])
    clf.set_params(**params)
    results = cross_validate(clf, features, truth, cv=5, return_train_score=False, n_jobs=-1)
    run_time = timer() - start
    best_score = max(results["test_score"])
    loss = 1 - best_score
    return {"loss": loss, "status": STATUS_OK, "train_time": run_time, "params": params}


def evaluate(clf, features, truth):
    results = cross_validate(clf, features, truth, cv=5)
    print(results['train_score'])
    print(results['test_score'])


def optimize(space, clf, features, truth, int_variables, max_evals=300):
    bayes_trials = Trials()
    fmin(fn=partial(objective, clf, features, truth, int_variables), space=space, algo=tpe.suggest, max_evals=max_evals,
         trials=bayes_trials)
    bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
    print(1 - bayes_trials_results[0]["loss"])

def plotlearningcurve(clf, features, truth):
    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, features, truth)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.show()
    
