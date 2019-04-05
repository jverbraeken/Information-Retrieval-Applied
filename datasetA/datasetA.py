import codecs
import json
import os
import pickle
from typing import List, Dict, Tuple

import enchant
import nltk
import textstat
from hyperopt import hp
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import ml_util

dictionary = enchant.Dict("en_US")
sub_datasets = ["datasetA1"]
sentiment_analyzer = SentimentIntensityAnalyzer()
with open("contractions.txt", 'r') as file:
    contractions = list(map(lambda x: x.replace('\n', ''), file.readlines()))



def _parse_json(file1: str, file2: str) -> Tuple[List[Dict], List[Dict]]:
    with codecs.open(file1, encoding='utf8') as f:
        instances = json.load(f)

    with codecs.open(file2, encoding='utf8') as f:
        truth = json.load(f)

    return instances, truth


def _extract_features(dataset: List[Dict]) -> List[Dict]:
    sentiment_analyzer = SentimentIntensityAnalyzer()
    with open("contractions.txt", 'r') as file:
        contractions = list(map(lambda x: x.replace('\n', ''), file.readlines()))
    result = []

    for i, item in enumerate(dataset):
        if i % 50 == 0:
            print(str(float(i) * 100 / float(len(dataset))) + "%")
        num_characters_post_title = sum([len(x) for x in item['postText']])
        num_characters_article_title = len(item['targetTitle'])
        num_characters_article_description = len(item['targetDescription'])
        num_characters_article_keywords = len(item['targetKeywords'])
        num_characters_article_captions = sum([len(x) for x in item['targetCaptions']])
        num_characters_article_paragraphs = sum([len(x) for x in item['targetParagraphs']])

        num_formal_words = sum(
            [0 if x == "" else dictionary.check(x) for y in item['targetParagraphs'] for x in y.split(' ')])
        total_words = len([x for y in item['targetParagraphs'] for x in y.split(' ')])

        sentiment_post_title = sentiment_analyzer.polarity_scores(item['postText'][0])['compound']
        sentiment_article_title = sentiment_analyzer.polarity_scores(item['targetTitle'][0])['compound']
        sentiment_article_paragraphs = sum(
            sentiment_analyzer.polarity_scores(x)['compound'] for x in item['targetParagraphs'])

        readability_article_paragraphs = None
        article_paragraph = ' '.join(item['targetParagraphs'])
        if len(article_paragraph.split()) >= 100:
            readability_article_paragraphs = textstat.flesch_kincaid_grade(article_paragraph)

        starts_with_number_post_title = 0 if len(item['postText'][0]) == 0 else item['postText'][0][0].isdigit()
        number_of_dots_post_title = 0 if len(item['postText'][0]) == 0 else item['postText'][0][0].count('.')

        first_title_word = item['targetTitle'].split()[0].lower()

        result.append({
            "num_characters_post_title": num_characters_post_title,
            "num_characters_article_title": num_characters_article_title,
            "num_characters_article_description": num_characters_article_description,
            "num_characters_article_keywords": num_characters_article_keywords,
            "num_characters_article_captions": num_characters_article_captions,
            "num_characters_article_paragraphs": num_characters_article_paragraphs,

            "diff_num_characters_post_title_article_title": abs(
                num_characters_post_title - num_characters_article_title),
            "diff_num_characters_post_title_article_description": abs(
                num_characters_post_title - num_characters_article_description),
            "diff_num_characters_post_title_article_keywords": abs(
                num_characters_post_title - num_characters_article_keywords),
            "diff_num_characters_post_title_article_captions": abs(
                num_characters_post_title - num_characters_article_captions),
            "diff_num_characters_post_title_article_paragraphs": abs(
                num_characters_post_title - num_characters_article_paragraphs),

            "diff_num_characters_article_title_article_description": abs(
                num_characters_article_title - num_characters_article_description),
            "diff_num_characters_article_title_article_keywords": abs(
                num_characters_article_title - num_characters_article_keywords),
            "diff_num_characters_article_title_article_captions": abs(
                num_characters_article_title - num_characters_article_captions),
            "diff_num_characters_article_title_article_paragraphs": abs(
                num_characters_article_title - num_characters_article_paragraphs),

            "diff_num_characters_article_description_article_keywords": abs(
                num_characters_article_description - num_characters_article_keywords),
            "diff_num_characters_article_description_article_captions": abs(
                num_characters_article_description - num_characters_article_captions),
            "diff_num_characters_article_description_article_paragraphs": abs(
                num_characters_article_description - num_characters_article_paragraphs),

            "diff_num_characters_article_keywords_article_captions": abs(
                num_characters_article_keywords - num_characters_article_captions),
            "diff_num_characters_article_keywords_article_paragraphs": abs(
                num_characters_article_keywords - num_characters_article_paragraphs),

            "diff_num_characters_article_captions_article_paragraphs": abs(
                num_characters_article_captions - num_characters_article_paragraphs),

            "ratio_num_characters_post_title_article_title": None if num_characters_article_title == 0 else num_characters_post_title / num_characters_article_title,
            "ratio_num_characters_post_title_article_description": None if num_characters_article_description == 0 else num_characters_post_title / num_characters_article_description,
            "ratio_num_characters_post_title_article_keywords": None if num_characters_article_keywords == 0 else num_characters_post_title / num_characters_article_keywords,
            "ratio_num_characters_post_title_article_captions": None if num_characters_article_captions == 0 else num_characters_post_title / num_characters_article_captions,
            "ratio_num_characters_post_title_article_paragraphs": None if num_characters_article_paragraphs == 0 else num_characters_post_title / num_characters_article_paragraphs,

            "ratio_num_characters_article_title_article_description": None if num_characters_article_description == 0 else num_characters_article_title / num_characters_article_description,
            "ratio_num_characters_article_title_article_keywords": None if num_characters_article_keywords == 0 else num_characters_article_title / num_characters_article_keywords,
            "ratio_num_characters_article_title_article_captions": None if num_characters_article_captions == 0 else num_characters_article_title / num_characters_article_captions,
            "ratio_num_characters_article_title_article_paragraphs": None if num_characters_article_paragraphs == 0 else num_characters_article_title / num_characters_article_paragraphs,

            "ratio_num_characters_article_description_article_keywords": None if num_characters_article_keywords == 0 else num_characters_article_description / num_characters_article_keywords,
            "ratio_num_characters_article_description_article_captions": None if num_characters_article_captions == 0 else num_characters_article_description / num_characters_article_captions,
            "ratio_num_characters_article_description_article_paragraphs": None if num_characters_article_paragraphs == 0 else num_characters_article_description / num_characters_article_paragraphs,

            "ratio_num_characters_article_keywords_article_captions": None if num_characters_article_captions == 0 else num_characters_article_keywords / num_characters_article_captions,
            "ratio_num_characters_article_keywords_article_paragraphs": None if num_characters_article_paragraphs == 0 else num_characters_article_keywords / num_characters_article_paragraphs,

            "ratio_num_characters_article_captions_article_paragraphs": None if num_characters_article_paragraphs == 0 else num_characters_article_captions / num_characters_article_paragraphs,

            "num_common_words_article_keywords_post_title": None if len(item['postText']) == 0 else sum(
                [item['postText'][0].count(x) for x in item['targetKeywords'].split(', ')]),
            "num_common_words_article_keywords_article_description": None if len(
                item['targetDescription']) == 0 else sum(
                [item['targetDescription'][0].count(x) for x in item['targetKeywords'].split(', ')]),
            "num_common_words_article_keywords_article_captions": None if len(item['targetCaptions']) == 0 else sum(
                [item['targetCaptions'][0].count(x) for x in item['targetKeywords'].split(', ')]),
            "num_common_words_article_keywords_article_paragraphs": None if len(item['targetParagraphs']) == 0 else sum(
                [item['targetParagraphs'][0].count(x) for x in item['targetKeywords'].split(', ')]),

            "num_formal_words": num_formal_words,
            "num_informal_words": total_words - num_formal_words,

            "ratio_formal_words": None if total_words == 0 else num_formal_words / total_words,
            "ratio_informal_words": None if total_words == 0 else 1 - num_formal_words / total_words,

            "sentiment_post_title": sentiment_post_title,
            "sentiment_article_title": sentiment_article_title,
            "sentiment_article_paragraphs": sentiment_article_paragraphs,

            "readability_article_paragraphs": readability_article_paragraphs,

            "starts_with_number_post_title": starts_with_number_post_title,
            "number_of_dots_post_title": number_of_dots_post_title,

            "has_demonstratives": first_title_word in {"this", "that", "these", "those"},
            "has_third_pronoun": first_title_word in {"he", "she", "it", "his", "her", "its", "him"},
            "has_definitive": first_title_word in {"the", "a", "an"},
            "is_start_adverb": nltk.pos_tag(first_title_word)[0][1] == "RB",

            "num_contractions": len(list(filter(lambda x: x in contractions, item["targetTitle"].split()))),
        })
    return result


def _extract_truth_labels(dataset: List[Dict]) -> List[Dict]:
    result = []
    for entry in dataset:
        result.append(entry["truthClass"] == "clickbait")
    return result


def _extract_and_store_features(name: str, dataset: List[Dict]) -> None:
    pickle_name = os.path.join("generated", name + "-features.pickle")
    print(pickle_name)

    if os.path.isfile(pickle_name):
        with open(pickle_name, 'rb') as f:
            result = pickle.load(f)
    else:
        result = _extract_features(dataset)
        with open(pickle_name, 'wb') as f:
            pickle.dump(result, f)

    # return result


def _extract_and_store_truth_labels(name: str, dataset: List[Dict]) -> None:
    pickle_name = os.path.join("generated", name + "-truth.pickle")
    print(pickle_name)
    if os.path.isfile(pickle_name):
        with open(pickle_name, 'rb') as f:
            result = pickle.load(f)
    else:
        result = _extract_truth_labels(dataset)
        with open(pickle_name, 'wb') as f:
            pickle.dump(result, f)

    # return result


def get_and_store_features() -> None:
    instances = []
    truths = []

    for sub_dataset in sub_datasets:
        instance, truth = _parse_json(os.path.join("datasetA", sub_dataset, "instances.jsonl"),
                                      os.path.join("datasetA", sub_dataset, "truth.jsonl"))
        instances.append(instance)
        truths.append(truth)

    for sub_dataset, instance in zip(sub_datasets, instances):
        _extract_and_store_features(sub_dataset, instance)

    for sub_dataset, truth in zip(sub_datasets, truths):
        _extract_and_store_truth_labels(sub_dataset, truth)


def _load_features_truth(normalization, pca) -> Tuple[List, List]:
    name = sub_datasets[0]
    pickle_name_features = os.path.join("generated", name + "-features.pickle")
    pickle_name_truth = os.path.join("generated", name + "-truth.pickle")
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


def svc_RFE(normalization) -> None:
    features, truth = _load_features_truth(normalization, False)
    svc = SVC(verbose=True, kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=100, step=100)
    rfe.fit(features, truth)
    ranking = rfe.ranking_
    print(ranking)
    # ranking = rfe.ranking_.reshape(digits.images[0].shape)


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
