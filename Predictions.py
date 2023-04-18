import numpy as np
import tensorflow as tf
from transformers import BertTokenizer


class Predictions:
    def __init__(self, model: tf.Module, tokenizer: BertTokenizer):
        """
        Initializes the class with model and tokenizer values.
        model: The Pretrained model.
        tokenizer: tokenizer with pretrained weights.
        """
        self.model = model
        self.tokenizer = tokenizer

    def preprocess_input(self, question: str, context: str, tokenizer: BertTokenizer, sequence_length: int):
        """
        Preprocesses the data and converts them into formats of input_ids, attention_masks, token_type_ids.

        ARGS:
            question: question which needs to be processed.
            context: context that needs to be processed.
            tokenizer: pretrained tokenizer with weights of pretrained model.
            sequence_length: sequence_length of each word.

        Returns:
            returns a tuple of input_ids, attention_mask, token_type_ids.
        """

        encoded_data = tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=sequence_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="tf",
        )

        input_ids = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]
        token_type_ids = encoded_data["token_type_ids"]
        return input_ids, attention_mask, token_type_ids

    def get_answer(self, context: str, start_pos: int, end_pos: int, tokenizer: BertTokenizer):
        """
        Extracts the final predicted answer based on start and end idx.

        ARGS:
            context: The context in which answer is present.
            start_pos: Start index of the answer in context.
            end_pos: end index of the answer in context.

        Returns:
            returns the final predicted answer.
        """
        context_tokens = tokenizer.tokenize(context)
        answer_tokens = context_tokens[start_pos:end_pos+1]
        answer_text = tokenizer.convert_tokens_to_string(answer_tokens)

        return answer_text

    def make_prediction(self, question: str, context: str, sequence_length: int = 384):
        """
        Displays the final answer by taking input question and context as parameter.

        ARGS:
            question: Input question.
            context: Input context.
            sequence_length: Max sequence to be taken for each word.

        Returns:
            returns the final answer if found.
        """
        input_ids, attention_mask, token_type_ids = self.preprocess_input(
            question, context, self.tokenizer, sequence_length)
        start_logits, end_logits = self.model.predict(
            [input_ids, attention_mask])

        start_position = np.argmax(start_logits)
        end_position = np.argmax(end_logits)

        # Check if the end position is greater than or equal to the start position, otherwise return an empty string
        if end_position >= start_position:
            answer = self.get_answer(context, start_position,
                                     end_position, self.tokenizer)
        else:
            answer = ""

        return answer
