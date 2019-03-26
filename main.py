from typing import Dict, List


def extract_features(dataset: List) -> List[Dict]:
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


def main():



if __name__ == "__main__":
    main()
