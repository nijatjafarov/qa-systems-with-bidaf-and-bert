import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em_f1(pred, truth):
    pred_norm = normalize_answer(pred)
    truth_norm = normalize_answer(truth)
    em = int(pred_norm == truth_norm)
    pred_tokens = pred_norm.split()
    truth_tokens = truth_norm.split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        f1 = int(pred_tokens == truth_tokens)
    else:
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(truth_tokens)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return em, f1