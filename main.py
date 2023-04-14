"""
Jyothi Vishnu Vardhan Kolla.

This is the main file which performs tasks such as 
exporataroy data anaysis,..., based on command line inputs.
"""
import sys
from Exploratory_data_analysis import ExploratoryAnalysis
from Data_prep import PrepareData


def main(argv):
    """
        Main which takes command line arguements and executes the pipeline.
        ARGS:
            argv[1]: if given-1 performs exploratory data analysis on given data.
            argv[2]: if given-1 preprocess and prepares the final data.
    """
    perform_eda = int(argv[1])
    prepare_data = int(argv[2])

    # Paths to data.
    train_tsv_path = "/Users/jyothivishnuvardhankolla/Desktop/SoftinWay/WikiQACorpus/WikiQA-train.tsv"
    dev_tsv_path = "/Users/jyothivishnuvardhankolla/Desktop/SoftinWay/WikiQACorpus/WikiQA-dev.tsv"
    test_tsv_path = "/Users/jyothivishnuvardhankolla/Desktop/SoftinWay/WikiQACorpus/WikiQA-test.tsv"

    if perform_eda == 1:  # If given 1 by user perfrom EDA.
        # Initialize the ExploratoryAnalysis object.
        ob = ExploratoryAnalysis(train_tsv=train_tsv_path, dev_tsv=dev_tsv_path,
                                 test_tsv=test_tsv_path)

        ob.exploreTrainTsv()  # Performs EDA for TrainTSV.
        ob.exploreDevTsv()  # Performs EDA for DevTSV.
        ob.exploreTestTsv()  # Performs EDA for TestTSV

    if prepare_data == 1:  # If given 1 by user prepares the data need to train the model.
        pos_ans_path = "/Users/jyothivishnuvardhankolla/Desktop/SoftinWay/WikiQACorpus/WikiQASent.pos.ans.tsv"
        # Initialize the PrepareData object.
        ob = PrepareData(train_tsv=train_tsv_path, dev_tsv=dev_tsv_path,
                         test_tsv=test_tsv_path, pos_ans_tsv=pos_ans_path)
        ob.Preprocess()


if __name__ == "__main__":
    main(sys.argv)
