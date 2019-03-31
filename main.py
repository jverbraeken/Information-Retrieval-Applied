from cmd import Cmd

from nltk import download

from datasetA import datasetA
from datasetB import datasetB


class Shell(Cmd):
    def install_dependencies(self, args):
        download('vader_lexicon')

    def do_extract_features(self, args):
        """Parses the dataset(s), extract the features, and stores them on the disk."""
        if len(args) == 0:
            for dataset in [datasetA, datasetB]:
                dataset.get_and_store_features()
            print("Stored features found in dataset A and B")
        elif args[0] == "A":
            datasetA.get_and_store_features()
            print("Stored features found in dataset A")
        elif args[0] == "B":
            datasetB.process()
            print("Stored features found in dataset B")

    def do_train_and_test(self, args):
        """Trains and tests a classifier on a dataset."""
        if len(args) == 0:
            print("Enter the name of the dataset")
        elif args[0] == "A":
            datasetA.train_and_test_svc()


def main() -> None:
    #  Dependencies
    shell = Shell()
    shell.prompt = "> "
    shell.cmdloop("Starting shell...")
    #  CLI
    # clickbaitB, nonclickbaitB = load("datasetB/clickbait_data.jsonl", "datasetB/non_clickbait_data.jsonl")
    # featuresA1 = datasetA.get_features("featuresA1", instanceA1)
    # featuresA2 = get_features("featuresA2", instanceA2)


if __name__ == "__main__":
    main()

# https://clickbait-detector.herokuapp.com/detect?headline=
