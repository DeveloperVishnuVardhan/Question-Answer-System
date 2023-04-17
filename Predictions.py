import numpy as np


class Predictions:
    def __init__(self, model, tokenizer):
        """
        """
        self.model = model
        self.tokenizer = tokenizer

    def preprocess_input(self, question, context, tokenizer, sequence_length):
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

    def get_answer(self, context, start_pos, end_pos, tokenizer):
        context_tokens = tokenizer.tokenize(context)
        answer_tokens = context_tokens[start_pos:end_pos+1]
        answer_text = tokenizer.convert_tokens_to_string(answer_tokens)

        return answer_text

    def make_prediction(self, question, context, sequence_length=384):
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
