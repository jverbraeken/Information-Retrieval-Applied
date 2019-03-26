import codecs
import json
from typing import Dict, List, Tuple
import requests

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn import datasets

def extract_features(dataset: List[Dict]) -> List[Dict]:
    result = []
    for item in dataset:
        result.append({
            "num_characters_post_title": 1,
            "num_characters_article_title": 1,
            "num_characters_article_description": 1,
            "num_characters_article_keywords": 1,
            "num_characters_article_captions": 1,
            "num_characters_article_paragraphs": 1,

            "diff_num_characters_post_title_article_title": 1,
            "diff_num_characters_post_title_article_description": 1,
            "diff_num_characters_post_title_article_keywords": 1,
            "diff_num_characters_post_title_article_captions": 1,
            "diff_num_characters_post_title_article_paragraphs": 1,

            "diff_num_characters_article_title_article_description": 1,
            "diff_num_characters_article_title_article_keywords": 1,
            "diff_num_characters_article_title_article_captions": 1,
            "diff_num_characters_article_title_article_paragraphs": 1,

            "diff_num_characters_article_description_article_keywords": 1,
            "diff_num_characters_article_description_article_captions": 1,
            "diff_num_characters_article_description_article_paragraphs": 1,

            "diff_num_characters_article_keywords_article_captions": 1,
            "diff_num_characters_article_keywords_article_paragraphs": 1,

            "diff_num_characters_article_captions_article_paragraphs": 1,

            "ratio_num_characters_post_title_article_title": 1,
            "ratio_num_characters_post_title_article_description": 1,
            "ratio_num_characters_post_title_article_keywords": 1,
            "ratio_num_characters_post_title_article_captions": 1,
            "ratio_num_characters_post_title_article_paragraphs": 1,

            "ratio_num_characters_article_title_article_description": 1,
            "ratio_num_characters_article_title_article_keywords": 1,
            "ratio_num_characters_article_title_article_captions": 1,
            "ratio_num_characters_article_title_article_paragraphs": 1,

            "ratio_num_characters_article_description_article_keywords": 1,
            "ratio_num_characters_article_description_article_captions": 1,
            "ratio_num_characters_article_description_article_paragraphs": 1,

            "ratio_num_characters_article_keywords_article_captions": 1,
            "ratio_num_characters_article_keywords_article_paragraphs": 1,

            "ratio_num_characters_article_captions_article_paragraphs": 1,

            "num_common_words_article_keywords_post_title": 1,
            "num_common_words_article_keywords_article_description": 1,
            "num_common_words_article_keywords_article_captions": 1,
            "num_common_words_article_keywords_article_paragraphs": 1,

            "num_formal_english_words": 1,
            "num_informal_english_words": 1,

            "ratio_formal_words": 1,
            "ratio_informal_words": 1,
        })
    return result


def main() -> None:
    instanceA1, truthA1 = parse("datasetA1/instances.jsonl", "datasetA1/truth.jsonl")
    instanceA2, truthA2 = parse("datasetA2/instances.jsonl", "datasetA2/truth.jsonl")
    clickbaitB, nonclickbaitB = parse("datasetB/clickbait_data.jsonl", "datasetB/non_clickbait_data.jsonl")

    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    iris = datasets.load_iris()
    #TODO insert predictions here!
    X, y = iris.data, iris.target
    clf.fit(X, y)
    cv_results = cross_validate(clf, X, y, cv=5, return_train_score = False)
    print(cv_results['test_score'])


    pass


def parse(file1: str, file2: str) -> Tuple[List[Dict], List[Dict]]:
    with codecs.open(file1, encoding='utf8') as f:
        instances = json.load(f)

    with codecs.open(file2, encoding='utf8') as f:
        truth = json.load(f)

    return instances, truth

if __name__ == "__main__":
    main()

# https://clickbait-detector.herokuapp.com/detect?headline=
