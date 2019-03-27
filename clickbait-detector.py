import codecs
import json
import requests
from main import parse

def main():
    instanceA1, truthA1 = parse("datasetA1/instances.jsonl", "datasetA1/truth.jsonl")
    instanceA2, truthA2 = parse("datasetA2/instances.jsonl", "datasetA2/truth.jsonl")
    clickbaitB, nonclickbaitB = parse("datasetB/clickbait_data.jsonl", "datasetB/non_clickbait_data.jsonl")

    clickbait_results = clickbaitDetector(nonclickbaitB)
    print(json.dumps(clickbait_results))
    with open("datasetB/clickbaitDetector-nonclickbait-results.json", "w") as f:
        f.write(json.dumps(clickbait_results))

    pass

def clickbaitDetector(data):
    results = []
    c = 0
    for d in data:
        failed = True
        while failed:
            try:
                resp = requests.get("https://clickbait-detector.herokuapp.com/detect?headline={}".format(d))
                res = resp.json()
                val= float(res["clickbaitiness"])/100.0
                print("{} {} ".format(c,val))
                c+=1
                results.append(val)
                failed = False
            except Exception as e:
                print(e)
    return results


if __name__ == "__main__":
    main()




# https://clickbait-detector.herokuapp.com/detect?headline=
