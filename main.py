import codecs
import json

def main():
    instanceA1, truthA1 = parse("datasetA1/instances.jsonl", "datasetA1/truth.jsonl")
    instanceA2, truthA2 = parse("datasetA2/instances.jsonl", "datasetA2/truth.jsonl")
    clickbaitB, nonclickbaitB = parse("datasetB/clickbait_data.jsonl", "datasetB/non_clickbait_data.jsonl")

    pass

def parse(file1, file2):
    with codecs.open(file1, encoding='utf8') as f:
        instances = json.load(f)

    with codecs.open(file2, encoding='utf8') as f:
        truth = json.load(f)

    return instances, truth


if __name__ == "__main__":
    main()




# https://clickbait-detector.herokuapp.com/detect?headline=