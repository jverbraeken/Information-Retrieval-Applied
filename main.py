import codecs
import json
from typing import Dict, List, Tuple
from PyDictionary import PyDictionary
import enchant

dictionary = enchant.Dict("en_US")


def extract_features(dataset: List[Dict]) -> List[Dict]:
    result = []


    for item in dataset:
        num_characters_post_title = sum([len(x) for x in item['postText']])
        num_characters_article_title = sum([len(x) for x in item['targetTitle']])
        num_characters_article_description = sum([len(x) for x in item['targetDescription']])
        num_characters_article_keywords = sum([len(x) for x in item['targetKeywords']])
        num_characters_article_captions = sum([len(x) for x in item['targetCaptions']])
        num_characters_article_paragraphs = sum([len(x) for x in item['targetParagraphs']])

        num_formal_words = sum([0 if x == "" else dictionary.check(x) for y in item['targetParagraphs'] for x in y.split(' ')])
        total_words = len([x for y in item['targetParagraphs'] for x in y.split(' ')])

        result.append({
            "num_characters_post_title": num_characters_post_title,
            "num_characters_article_title": num_characters_article_title,
            "num_characters_article_description": num_characters_article_description,
            "num_characters_article_keywords": num_characters_article_keywords,
            "num_characters_article_captions": num_characters_article_captions,
            "num_characters_article_paragraphs": num_characters_article_paragraphs,

            "diff_num_characters_post_title_article_title": abs(num_characters_post_title - num_characters_article_title),
            "diff_num_characters_post_title_article_description": abs(num_characters_post_title - num_characters_article_description),
            "diff_num_characters_post_title_article_keywords": abs(num_characters_post_title - num_characters_article_keywords),
            "diff_num_characters_post_title_article_captions": abs(num_characters_post_title - num_characters_article_captions),
            "diff_num_characters_post_title_article_paragraphs": abs(num_characters_post_title - num_characters_article_paragraphs),

            "diff_num_characters_article_title_article_description": abs(num_characters_article_title - num_characters_article_description),
            "diff_num_characters_article_title_article_keywords": abs(num_characters_article_title - num_characters_article_keywords),
            "diff_num_characters_article_title_article_captions": abs(num_characters_article_title - num_characters_article_captions),
            "diff_num_characters_article_title_article_paragraphs": abs(num_characters_article_title - num_characters_article_paragraphs),

            "diff_num_characters_article_description_article_keywords": abs(num_characters_article_description - num_characters_article_keywords),
            "diff_num_characters_article_description_article_captions": abs(num_characters_article_description - num_characters_article_captions),
            "diff_num_characters_article_description_article_paragraphs": abs(num_characters_article_description - num_characters_article_paragraphs),

            "diff_num_characters_article_keywords_article_captions": abs(num_characters_article_keywords - num_characters_article_captions),
            "diff_num_characters_article_keywords_article_paragraphs": abs(num_characters_article_keywords - num_characters_article_paragraphs),

            "diff_num_characters_article_captions_article_paragraphs": abs(num_characters_article_captions - num_characters_article_paragraphs),

            "ratio_num_characters_post_title_article_title": 0 if num_characters_article_title == 0 else num_characters_post_title / num_characters_article_title,
            "ratio_num_characters_post_title_article_description": 0 if num_characters_article_description == 0 else num_characters_post_title / num_characters_article_description,
            "ratio_num_characters_post_title_article_keywords": 0 if num_characters_article_keywords == 0 else num_characters_post_title / num_characters_article_keywords,
            "ratio_num_characters_post_title_article_captions": 0 if num_characters_article_captions == 0 else num_characters_post_title / num_characters_article_captions,
            "ratio_num_characters_post_title_article_paragraphs": 0 if num_characters_article_paragraphs == 0 else num_characters_post_title / num_characters_article_paragraphs,

            "ratio_num_characters_article_title_article_description": 0 if num_characters_article_description == 0 else num_characters_article_title / num_characters_article_description,
            "ratio_num_characters_article_title_article_keywords": 0 if num_characters_article_keywords == 0 else num_characters_article_title / num_characters_article_keywords,
            "ratio_num_characters_article_title_article_captions": 0 if num_characters_article_captions == 0 else num_characters_article_title / num_characters_article_captions,
            "ratio_num_characters_article_title_article_paragraphs": 0 if num_characters_article_paragraphs == 0 else num_characters_article_title / num_characters_article_paragraphs,

            "ratio_num_characters_article_description_article_keywords": 0 if num_characters_article_keywords == 0 else num_characters_article_description / num_characters_article_keywords,
            "ratio_num_characters_article_description_article_captions": 0 if num_characters_article_captions == 0 else num_characters_article_description / num_characters_article_captions,
            "ratio_num_characters_article_description_article_paragraphs": 0 if num_characters_article_paragraphs == 0 else num_characters_article_description / num_characters_article_paragraphs,

            "ratio_num_characters_article_keywords_article_captions": 0 if num_characters_article_captions == 0 else num_characters_article_keywords / num_characters_article_captions,
            "ratio_num_characters_article_keywords_article_paragraphs": 0 if num_characters_article_paragraphs == 0 else num_characters_article_keywords / num_characters_article_paragraphs,

            "ratio_num_characters_article_captions_article_paragraphs": 0 if num_characters_article_paragraphs == 0 else num_characters_article_captions / num_characters_article_paragraphs,

            "num_common_words_article_keywords_post_title": 0 if len(item['postText']) == 0 else sum([item['postText'][0].count(x) for x in item['targetKeywords'].split(', ')]),
            "num_common_words_article_keywords_article_description": 0 if len(item['targetDescription']) == 0 else sum([item['targetDescription'][0].count(x) for x in item['targetKeywords'].split(', ')]),
            "num_common_words_article_keywords_article_captions": 0 if len(item['targetCaptions']) == 0 else sum([item['targetCaptions'][0].count(x) for x in item['targetKeywords'].split(', ')]),
            "num_common_words_article_keywords_article_paragraphs": 0 if len(item['targetParagraphs']) == 0 else sum([item['targetParagraphs'][0].count(x) for x in item['targetKeywords'].split(', ')]),

            "num_formal_words": num_formal_words,
            "num_informal_words": total_words - num_formal_words,

            "ratio_formal_words": num_formal_words / total_words,
            "ratio_informal_words": 1 - num_formal_words / total_words
        })
    return result


def main() -> None:
    instanceA1, truthA1 = parse("datasetA1/instances.jsonl", "datasetA1/truth.jsonl")
    instanceA2, truthA2 = parse("datasetA2/instances.jsonl", "datasetA2/truth.jsonl")
    clickbaitB, nonclickbaitB = parse("datasetB/clickbait_data.jsonl", "datasetB/non_clickbait_data.jsonl")
    featuresA1 = extract_features(instanceA1)


def parse(file1: str, file2: str) -> Tuple[List[Dict], List[Dict]]:
    with codecs.open(file1, encoding='utf8') as f:
        instances = json.load(f)

    with codecs.open(file2, encoding='utf8') as f:
        truth = json.load(f)

    return instances, truth

if __name__ == "__main__":
    main()

# https://clickbait-detector.herokuapp.com/detect?headline=
