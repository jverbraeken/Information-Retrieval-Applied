import codecs
import json
import os
import pickle
from typing import List, Dict, Tuple

import enchant
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer

dictionary = enchant.Dict("en_US")


def _extract_features(dataset: List[Dict]) -> List[Dict]:
    # TODO
    return []


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


def process() -> List[List[Dict]]:
    instances = []
    truths = []

    # for sub_dataset in sub_datasets:
    #     instance, truth = _parse_json(os.path.join(sub_dataset, "instances.jsonl"), os.path.join(sub_dataset, "truth.jsonl"))
    #     instances.append(instance)
    #     truths.append(truth)
    #
    # features = []
    # for sub_dataset, instance in zip(sub_datasets, instances):
    #     features.append(_get_and_store_features(sub_dataset, instance))

    return []
