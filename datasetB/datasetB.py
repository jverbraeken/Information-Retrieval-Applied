import codecs
import json
import os
import pickle
from typing import List, Dict, Tuple

import enchant
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import textstat

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

        num_formal_words = sum(
            [0 if x == "" else dictionary.check(x) for y in item for x in y.split(' ')])
        total_words = len(item.split())

        sentiment = sentiment_analyzer.polarity_scores(item)

        starts_with_number = item[0].isdigit()
        number_of_dots = item[0].count('.')

        first_title_word = item.split()[0].lower()

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
            "is_start_adverb": nltk.pos_tag(first_title_word)[0][1] == "RB",

            "num_contractions": len(list(filter(lambda x: x in contractions, item.split()))),
        })
    return result


def _get_and_store_features(name: str, dataset: List[Dict]) -> List[Dict]:
    pickle_name = name + ".pickle"

    if os.path.isfile(pickle_name):
        with open(pickle_name, 'rb') as f:
            result = pickle.load(f)
    else:
        result = _extract_features(dataset)
        with open(pickle_name, 'wb') as f:
            pickle.dump(result, f)

    return result


def _parse_json(file1: str, file2: str) -> Tuple[List[Dict], List[Dict]]:
    with codecs.open(file1, encoding='utf8') as f:
        instances = json.load(f)

    with codecs.open(file2, encoding='utf8') as f:
        truth = json.load(f)

    return instances, truth


def get_and_store_features() -> None:
    clickbait, non_clickbait = _parse_json(os.path.join("datasetB", "clickbait_data.jsonl"),
                                           os.path.join("datasetB", "non_clickbait_data.jsonl"))

    instance = clickbait.extend(non_clickbait)
    truth = [1] * len(clickbait)
    truth.extend([0] * len(non_clickbait))
    _extract_and_store_features(instance)
    _extract_and_store_truth_labels(truth)
