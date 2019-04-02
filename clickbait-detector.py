import codecs
import json
import requests
from typing import Dict, List, Tuple

def main():
    instanceA1, truthA1 = load("datasetA1/instances.jsonl", "datasetA1/truth.jsonl")
    instanceA2, truthA2 = load("datasetA2/instances.jsonl", "datasetA2/truth.jsonl")
    clickbaitB, nonclickbaitB = load("datasetB/clickbait_data.jsonl", "datasetB/non_clickbait_data.jsonl")

    # clickbait_results = clickbaitDetectorB(nonclickbaitB)
    # print(json.dumps(clickbait_results))
    # with open("datasetB/clickbaitDetector-nonclickbait-results.json", "w") as f:
    #     f.write(json.dumps(clickbait_results))
    #
    # clickbait_results = clickbaitDetectorB(clickbaitB)
    # print(json.dumps(clickbait_results))
    # with open("datasetB/clickbaitDetector-clickbait-results.json", "w") as f:
    #     f.write(json.dumps(clickbait_results))

    clickbait_results = clickbaitDetectorA(instanceA1)
    print(json.dumps(clickbait_results))
    with open("datasetA1/clickbaitDetector-results.json", "w") as f:
        f.write(json.dumps(clickbait_results))

    clickbait_results = clickbaitDetectorA(instanceA2)
    print(json.dumps(clickbait_results))
    with open("datasetA2/clickbaitDetector-results.json", "w") as f:
        f.write(json.dumps(clickbait_results))

    pass


def clickbaitDetectorA(data):
    results = []
    c = 0
    for d in data:
        failed = True
        while failed:
            try:
                resp = requests.get("https://clickbait-detector.herokuapp.com/detect?headline={}".format(d["postText"][0]), timeout=5)
                res = resp.json()
                val= float(res["clickbaitiness"])/100.0
                print("{} {} ".format(c,val))
                c+=1
                results.append(val)
                failed = False
            except Exception as e:
                print(e)
    return results


def clickbaitDetectorB(data):
    results = []
    c = 0
    for d in data:
        failed = True
        while failed:
            try:
                resp = requests.get("https://clickbait-detector.herokuapp.com/detect?headline={}".format(d), timeout=5)
                res = resp.json()
                val= float(res["clickbaitiness"])/100.0
                print("{} {} ".format(c,val))
                c+=1
                results.append(val)
                failed = False
            except Exception as e:
                print(e)
    return results


def parse(file1: str, file2: str) -> Tuple[List[Dict], List[Dict]]:
    with codecs.open(file1, encoding='utf8') as f:
        instances = json.load(f)

    with codecs.open(file2, encoding='utf8') as f:
        truth = json.load(f)

    return instances, truth

if __name__ == "__main__":
    main()




# https://clickbait-detector.herokuapp.com/detect?headline=
