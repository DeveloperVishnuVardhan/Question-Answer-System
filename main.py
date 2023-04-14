"""
Jyothi Vishnu Vardhan Kolla.

This is the main file which performs tasks such as 
exporataroy data anaysis,..., based on command line inputs.
"""
import sys
from Exploratory_data_analysis import ExploratoryAnalysis



def main(argv):
    """
        Main which takes command line arguements and executes the pipeline.

        ARGS:

    """
    perform_eda = int(argv[1])
    if perform_eda == 1:  # If given 1 by user perfrom EDA.
        train_tsv_path = "/Users/jyothivishnuvardhankolla/Desktop/SoftinWay/WikiQACorpus/WikiQA-train.tsv"
        dev_tsv_path = "/Users/jyothivishnuvardhankolla/Desktop/SoftinWay/WikiQACorpus/WikiQA-dev.tsv"
        test_tsv_path = "/Users/jyothivishnuvardhankolla/Desktop/SoftinWay/WikiQACorpus/WikiQA-test.tsv"

        # Initialize the ExploratoryAnalysis object.
        ob = ExploratoryAnalysis(train_tsv=train_tsv_path, dev_tsv=dev_tsv_path,
                            test_tsv=test_tsv_path)
        
        ob.exploreTrainTsv() # Performs EDA for TrainTSV.
        ob.exploreDevTsv() # Performs EDA for DevTSV.
        ob.exploreTestTsv() # Performs EDA for TestTSV


if __name__ == "__main__":
    main(sys.argv)