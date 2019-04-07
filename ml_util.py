from functools import partial
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, Trials
from sklearn.model_selection import cross_validate, learning_curve, train_test_split
from sklearn.metrics import classification_report, roc_auc_score


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

def metrics(clf, features, truth):
    print('Calculating metrics 5 times, each time with different train test...')
    for x in range(0, 5):
        X_train, X_test, y_train, y_test = train_test_split(features, truth, test_size=0.2)
        clf.fit(X_train,y_train)
        preds = clf.predict(X_test)
        print(classification_report(y_test, preds))

        y_test_bin = []
        preds_bin = []
        for i in range(0, len(y_test)):
            if y_test[i] == 'clickbait':
                y_test_bin.append(1)
            else:
                y_test_bin.append(0)
            if preds[i] == 'clickbait':
                preds_bin.append(1)
            else:
                preds_bin.append(0)
                
        print('ROC AUC score:', roc_auc_score(y_test_bin, preds_bin))


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
    
