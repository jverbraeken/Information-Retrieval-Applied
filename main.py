from cmd import Cmd

from nltk import download

from datasetA import datasetA
from datasetB import datasetB


class Shell(Cmd):
    def do_install_dependencies(self, _):
        """Install dependencies for the libraries."""
        download('vader_lexicon')

    def do_extract_features(self, args):
        """Parses the dataset(s), extract the features, and stores them on the disk:    extract_features <A or B>"""
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
        """Trains and tests a classifier on a dataset:    train_and_test <A or B> <svc or random_forest>"""
        if len(args) == 0:
            print("Enter the name of the dataset")
        elif args[0] == "A":
            word = args.split()[1]
            if word == "svc":
                datasetA.train_and_test_svc()
            elif word == "random_forest":
                datasetA.train_and_test_random_forest()

    def do_quit(self, _):
        """Quits the program"""
        return True


def main() -> None:
    #  Dependencies
    shell = Shell()
    shell.prompt = "> "
    shell.cmdloop("Starting shell...")


if __name__ == "__main__":
    main()

# https://clickbait-detector.herokuapp.com/detect?headline=
