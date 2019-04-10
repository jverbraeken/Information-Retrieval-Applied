from cmd import Cmd

from nltk import download

from datasetA import datasetA
from datasetB import datasetB

train_and_test_methods_A = {
    "svc": lambda normalization, optimization, pca: datasetA.train_and_test_svc(normalization, optimization, pca),
    "random_forest": lambda normalization, optimization, pca: datasetA.train_and_test_random_forest(normalization,
                                                                                                    optimization, pca),
    "knn": lambda normalization, optimization, pca: datasetA.train_and_test_knn(normalization, optimization, pca),
}

train_and_test_methods_B = {
    "svc": lambda normalization, optimization, pca: datasetB.train_and_test_svc(normalization, optimization, pca),
    "random_forest": lambda normalization, optimization, pca: datasetB.train_and_test_random_forest(normalization,
                                                                                                    optimization, pca),
    "knn": lambda normalization, optimization, pca: datasetB.train_and_test_knn(normalization, optimization, pca),
}


class Shell(Cmd):
    def do_install_dependencies(self, _):
        """Install dependencies for the libraries."""
        download('vader_lexicon')
        download('averaged_perceptron_tagger')

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
            datasetB.get_and_store_features()
            print("Stored features found in dataset B")

    def do_train_and_test(self, args):
        """Trains and tests a classifier on a dataset:    train_and_test <A or B> <svc or random_forest>"""
        words = args.split()
        if len(words) == 0:
            print("Enter the name of the dataset")
        elif words[0] == "A":
            no_normalization = "--no_normalization" in words
            optimization = "--optimize" in words
            pca = "--pca" in words
            train_and_test_methods_A[words[1]](not no_normalization, optimization, pca)
        elif words[0] == "B":
            no_normalization = "--no_normalization" in words
            optimization = "--optimize" in words
            pca = "--pca" in words
            train_and_test_methods_B[words[1]](not no_normalization, optimization, pca)

    def do_rfe(self, args):
        words = args.split()
        if len(words) == 0:
            print("Enter the name of the dataset")
        elif words[0] == "A":
            no_normalization = words[2] == "--no_normalization"
            if words[1] == "svc":
                datasetA.svc_RFE(not no_normalization)

    def do_pca(self, args):
        words = args.split()
        if words[0] == "A":
            datasetA.pca()

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
