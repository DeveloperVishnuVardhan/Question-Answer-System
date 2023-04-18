"""
Jyothi Vishnu Vardhan Kolla.

This is the main file which performs tasks such as
exporataroy data anaysis,data-preparation,model-training based on command line inputs.
"""
import sys
from Exploratory_data_analysis import ExploratoryAnalysis
from Data_prep import PrepareData
from models import Models
import pandas as pd
from transformers import BertConfig, BertTokenizer
from utils import load_model
from Predictions import Predictions, Evaluations


def main(argv):
    """
        Main which takes command line arguements and executes the pipeline.

        ARGS:
            argv[1]: if given-1 performs exploratory data analysis on given data.
            argv[2]: if given-1 performing preprocessing and prepares the final data.
            argv[3]: if given-1 performs the training.
            argv[4]: if given-1 performs predictions based on given inputs.
    """
    perform_eda = int(argv[1])
    prepare_data = int(argv[2])
    train_mode = int(argv[3])
    prediction_mode = int(argv[4])
    evaluation_mode = int(argv[5])

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

    if train_mode == 1:  # If given 1 by user the model training begins.
        train_df = pd.read_csv("data/train.csv")
        dev_df = pd.read_csv("data/dev.csv")
        test_df = pd.read_csv("data/test.csv")
        ob = Models(train_df, dev_df, test_df)
        ob.train_model()

    if prediction_mode == 1:  # If given 1 by user prediction mode turns on.
        loaded_model = load_model("Models")
        tokenizer = BertTokenizer.from_pretrained("Models")
        ob = Predictions(loaded_model, tokenizer)
        question = "How will Diaphragm Pump work"
        context = "A diaphragm pump (also known as a Membrane pump, Air Operated Double Diaphragm Pump (AODD) or Pneumatic Diaphragm Pump) is a positive displacement pump that uses a combination of the reciprocating action of a rubber , thermoplastic or teflon diaphragm and suitable valves either side of the diaphragm ( check valve , butterfly valves, flap valves, or any other form of shut-off valves) to pump a fluid ."
        # Make a prediction using the loaded model and tokenizer
        answer = ob.make_prediction(question, context)
        print("Answer:", answer)

    if evaluation_mode == 1:  # If given 1 by user computes and displays the metrics.
        loaded_model = load_model("Models")
        tokenizer = BertTokenizer.from_pretrained("Models")
        train_df = pd.read_csv("data/train.csv")
        dev_df = pd.read_csv("data/dev.csv")
        test_df = pd.read_csv("data/test.csv")
        ob1 = Predictions(loaded_model, tokenizer)  # Predictions object.
        # Evaluations object.
        ob2 = Evaluations(train_df, dev_df, test_df,
                          loaded_model, tokenizer, ob1)
        ob2.compute_store_predictions()
        pr, re, f1 = ob2.display_metrics(pd.read_csv(
            "/Users/jyothivishnuvardhankolla/Desktop/SoftinWay/Chatbot-development/data/predictions.csv"))
        print(f"Precision is {pr} and recall is {re} and f1-score is {f1}")


if __name__ == "__main__":
    main(sys.argv)
