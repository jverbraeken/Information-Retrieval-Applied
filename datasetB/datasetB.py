import codecs
import json
import os
import pickle
from typing import List, Dict, Tuple

import enchant
import nltk
from hyperopt import hp
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import ml_util

dictionary = enchant.Dict("en_US")


def _extract_features(dataset: List[str]) -> List[Dict]:
    sentiment_analyzer = SentimentIntensityAnalyzer()
    with open("contractions.txt", 'r') as file:
        contractions = list(map(lambda x: x.replace('\n', ''), file.readlines()))
    result = []

    for i, item in enumerate(dataset):
        if i % 50 == 0:
            print(str(float(i) * 100 / float(len(dataset))) + "%")
        num_characters = len(item)

        num_formal_words = sum([0 if x == "" else dictionary.check(x) for y in item for x in y.split(' ')])
        total_words = len(item.split())

        sentiment = sentiment_analyzer.polarity_scores(item)['compound']

        starts_with_number = 0 if len(item) == 0 else item[0].isdigit()
        number_of_dots = 0 if len(item) == 0 else item[0].count('.')

        first_title_word = "" if len(item) == 0 else item.split()[0].lower()

        result.append({
            "num_characters": num_characters,

            "num_formal_words": num_formal_words,
            "num_informal_words": total_words - num_formal_words,

            "ratio_formal_words": None if total_words == 0 else num_formal_words / total_words,
            "ratio_informal_words": None if total_words == 0 else 1 - num_formal_words / total_words,

            "sentiment_post_title": sentiment,

            "starts_with_number": starts_with_number,
            "number_of_dots": number_of_dots,

            "has_demonstratives": first_title_word in {"this", "that", "these", "those"},
            "has_third_pronoun": first_title_word in {"he", "she", "it", "his", "her", "its", "him"},
            "has_definitive": first_title_word in {"the", "a", "an"},
            "is_start_adverb": False if first_title_word == "" else nltk.pos_tag(first_title_word)[0][1] == "RB",

            "num_contractions": len(list(filter(lambda x: x in contractions, item.split()))),
        })
    return result


def _extract_truth_labels(dataset: List[Dict]) -> List[Dict]:
    result = []
    for entry in dataset:
        result.append(entry["truthClass"] == "clickbait")
    return result


def _extract_and_store_features(dataset: List[str]) -> None:
    pickle_name = os.path.join("generated", "B-features.pickle")

    result = _extract_features(dataset)
    with open(pickle_name, 'wb') as f:
        pickle.dump(result, f)


def _extract_and_store_truth(truth: List[str]) -> None:
    pickle_name = os.path.join("generated", "B-truth.pickle")

    with open(pickle_name, 'wb') as f:
        pickle.dump(truth, f)


def _parse_json(file1: str, file2: str) -> Tuple[List[str], List[str]]:
    with codecs.open(file1, encoding='utf8') as f:
        instances = json.load(f)

    with codecs.open(file2, encoding='utf8') as f:
        truth = json.load(f)

    return instances, truth


def get_and_store_features() -> None:
    clickbait, non_clickbait = _parse_json(os.path.join("datasetB", "clickbait_data.jsonl"),
                                           os.path.join("datasetB", "non_clickbait_data.jsonl"))

    truth = [1] * len(clickbait)
    clickbait.extend(non_clickbait)
    truth.extend([0] * len(non_clickbait))
    _extract_and_store_features(clickbait)
    _extract_and_store_truth(truth)


def _load_features_truth(normalization, pca) -> Tuple[List, List]:
    pickle_name_features = os.path.join("generated", "B-features.pickle")
    pickle_name_truth = os.path.join("generated", "B-truth.pickle")
    with open(pickle_name_features, 'rb') as f:
        features = pickle.load(f)
    with open(pickle_name_truth, 'rb') as f:
        truth = pickle.load(f)
    features = [[0 if b[1] is None else b[1] for b in a.items()] for a in features]

    for k in reversed(range(len(truth))):
        if sum(truth) * 2 >= len(truth):
            break
        if not truth[k]:
            del truth[k]
            del features[k]

    if normalization:
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features).tolist()

    if pca:
        pca_model = PCA(n_components=0.99, svd_solver='full')
        features = pca_model.fit_transform(features)
        print(pca_model.explained_variance_ratio_)
        print(pca_model.singular_values_)

    return features, truth


def train_and_test_svc(normalization, optimization, pca) -> None:
    features, truth = _load_features_truth(normalization, pca)

    clf = SVC(verbose=True)
    X_train, X_test, y_train, y_test = train_test_split(features, truth, test_size=0.2)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    if optimization:
        space = {
            "gamma": hp.uniform("gamma", 0.0001, 10.0),
            "C": hp.uniform("C", 0.01, 10.0),
        }
        ml_util.optimize(space, clf, features, truth, [], max_evals=100)
    else:
        ml_util.evaluate(clf, features, truth)


def train_and_test_random_forest(normalization, optimize, pca) -> None:
    features, truth = _load_features_truth(normalization, pca)

    clf = RandomForestClassifier(verbose=True)
    if optimize:
        space = {
            "n_estimators": hp.quniform("n_estimators", 1, 20, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "max_depth": hp.quniform("max_depth", 1, 8, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
        }
        ml_util.optimize(space, clf, features, truth, ["n_estimators", "max_depth", "min_samples_split"])
    else:
        ml_util.evaluate(clf, features, truth)


def train_and_test_knn(normalization, optimize, pca) -> None:
    features, truth = _load_features_truth(normalization, pca)

    clf = KNeighborsClassifier()
    if optimize:
        space = {
            "n_neighbors": hp.quniform("n_neighbors", 1, 10, 1),
            "p": hp.quniform("p", 1, 3, 1),
            "n_jobs": hp.quniform("n_jobs", -1, -1, 1),
        }
        ml_util.optimize(space, clf, features, truth, ["n_neighbors", "p", "n_jobs"], max_evals=30)
    else:
        ml_util.evaluate(clf, features, truth)
