from .utils import convert_examples_to_features, read_examples
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from ..builder import MULTIMODAL_PIPELINES


@MULTIMODAL_PIPELINES.register_module()
class BertTextTokenize(object):
    def __init__(self, query_len=40, bert_model_path=''):
        self.query_len = query_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    def __call__(self, results):
        phrase = results["text_info"]["text"]
        uniq_idx = results["text_info"]["uniq_id"]
        examples = read_examples(phrase, uniq_idx)
        features = convert_examples_to_features(examples=examples, seq_length=self.query_len,
                                                tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        results["word_id"] = np.array(word_id, dtype=int)
        results["word_mask"] = np.array(word_mask, dtype=int)
        results['parse_out'] = [phrase]
        results['uniq_id'] = uniq_idx
        return results
