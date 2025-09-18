import re
import string
from collections import Counter

def normalize_text(text: str) -> str:
    """
    Normalize text by lowercasing, removing punctuation, articles, 
    and extra whitespace to make comparison fair.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\b(a|an|the)\b", " ", text)        # Remove articles
    text = re.sub(r"\s+", " ", text).strip()           # Remove extra spaces
    return text


def compute_exact_match(prediction: str, ground_truth: str) -> int:
    """
    Compute Exact Match (EM) score between prediction and ground truth.
    Returns 1 if they match after normalization, else 0.
    """
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.
    Measures how many tokens overlap between the two strings.
    """
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
