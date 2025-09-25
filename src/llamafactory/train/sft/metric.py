from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_jieba_available, is_nltk_available, is_rouge_available


if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

if is_jieba_available():
    import jieba  # type: ignore

if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

if is_rouge_available():
    from rouge_chinese import Rouge

def extract_answer(response):
    if response.strip():  # Checks if the response is not empty and not just whitespace
        match = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()  # Return the matched letter in uppercase
    return None


def map2id(string):
    if string == "A":
        return 0
    elif string == "B":
        return 1
    elif string == "C":
        return 2
    elif string == "D":
        return 3
    else:
        return 4


def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        labelids = []
        preds = []

        for pred, label in zip(decoded_preds, decoded_labels):
            try:
                pred_answer = extract_answer(pred)
            except:
                pred_answer = None
            try:
                label_answer = extract_answer(label)
            except:
                label_answer = None

            
            labelids.append(map2id(label_answer))
            preds.append(map2id(pred_answer))

            hypothesis = list(pred_answer)
            reference = list(label_answer)

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        f1_acc = compute_metrics(labelids, preds)
        out = {k: float(np.mean(v)) for k, v in score_dict.items()}
        out.update(f1_acc)
        return out
