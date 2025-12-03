import math
import re
from collections import Counter
from typing import Dict, Any, Optional

import numpy as np

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


_gpt2_model = None
_gpt2_tokenizer = None


def _ensure_gpt2(model_name: str = 'gpt2'):
    global _gpt2_model, _gpt2_tokenizer
    if not _TRANSFORMERS_AVAILABLE:
        return False
    if _gpt2_model is None:
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        _gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        _gpt2_model.eval()
    return True


def gpt2_perplexity(text: str, model_name: str = 'gpt2', device: Optional[str] = None) -> float:
    """Compute approximate perplexity using GPT-2.

    Returns large float on failure or if transformers not available.
    """
    if not _ensure_gpt2(model_name):
        return float('nan')

    tokenizer = _gpt2_tokenizer
    model = _gpt2_model

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    enc = tokenizer(text, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)

    max_length = model.config.n_positions
    stride = max_length // 2

    nlls = []
    total_tokens = 0
    seq_len = input_ids.size(1)
    if seq_len == 0:
        return float('nan')

    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        if trg_len <= 0:
            continue

        input_ids_segment = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_segment.clone()
        # mask out tokens that belong to the context window instead of the current target span
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids_segment, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood.item())
        total_tokens += trg_len

    if not total_tokens:
        return float('nan')

    ppl = math.exp(sum(nlls) / total_tokens)
    return float(ppl)


def stylometric_features(text: str) -> Dict[str, float]:
    text = text.strip()
    words = re.findall(r"\w+", text)
    sentences = re.split(r'[.!?]+\s*', text)
    sentences = [s for s in sentences if s.strip()]

    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    avg_sentence_len = float(np.mean([len(re.findall(r"\w+", s)) for s in sentences])) if sentences else 0.0

    punct_count = len(re.findall(r"[.,;:\-()\[\]\"'!?]", text))
    punct_ratio = punct_count / max(1, len(text))

    # repetition: proportion of repeated 3-grams
    tokens = [w.lower() for w in words]
    trigrams = [' '.join(tokens[i:i+3]) for i in range(max(0, len(tokens)-2))]
    rep_rate = 0.0
    if trigrams:
        c = Counter(trigrams)
        repeats = sum(v-1 for v in c.values() if v > 1)
        rep_rate = repeats / len(trigrams)

    # stopword-ish short word ratio (simple heuristic)
    short_word_ratio = sum(1 for w in words if len(w) <= 3) / max(1, len(words))

    return {
        'avg_word_len': avg_word_len,
        'avg_sentence_len': avg_sentence_len,
        'punct_ratio': punct_ratio,
        'rep_rate': rep_rate,
        'short_word_ratio': short_word_ratio,
        'num_words': len(words),
        'num_sentences': len(sentences),
    }


def detect_ai(text: str, use_gpt2: bool = True) -> Dict[str, Any]:
    """Return a dict with keys: probability (0-1), label, breakdown.

    This is a heuristic detector combining perplexity (if available) and stylometrics.
    """
    text = text.strip()
    if not text:
        return {'probability': 0.0, 'label': 'Unclear', 'breakdown': {'reason': 'empty input'}}

    features = stylometric_features(text)
    ppx = float('nan')
    if use_gpt2 and _TRANSFORMERS_AVAILABLE:
        try:
            ppx = gpt2_perplexity(text)
        except Exception:
            ppx = float('nan')

    # Heuristic mapping: lower perplexity tends toward AI (not always true)
    score_parts = []

    # perplexity score
    if not math.isnan(ppx):
        # map ppx to 0..1 where lower ppx -> closer to 1 (AI)
        # clamp between 10 and 1000
        p = max(10.0, min(1000.0, ppx))
        ppx_score = (200.0 - p) / 200.0  # p=200 -> 0
        ppx_score = max(-1.0, min(1.0, ppx_score))
        ppx_score = (ppx_score + 1) / 2.0  # normalize to 0..1
        score_parts.append(('ppx', ppx_score, 0.6))
    else:
        # fallback: no ppx
        pass

    # repetition: higher repetition -> more likely AI
    rep = features['rep_rate']
    rep_score = min(1.0, rep * 5.0)
    score_parts.append(('rep', rep_score, 0.15))

    # short sentences & short word ratio: AI sometimes has more uniform short words
    short_ratio = features['short_word_ratio']
    short_score = max(0.0, (short_ratio - 0.35) / 0.35)
    short_score = min(1.0, short_score)
    score_parts.append(('short', short_score, 0.1))

    # punctuation: lower punctuation ratio -> more likely AI (heuristic)
    punct = features['punct_ratio']
    punct_score = 1.0 - min(0.05, punct) / 0.05 if punct >= 0 else 0.0
    punct_score = max(0.0, min(1.0, 1.0 - (punct / 0.03)))
    score_parts.append(('punct', punct_score, 0.15))

    # Combine weighted scores
    total_weight = sum(w for _, _, w in score_parts)
    combined = 0.0
    for name, s, w in score_parts:
        combined += s * w
    combined = combined / max(1e-9, total_weight)

    # Map combined to probability 0..1
    probability = float(max(0.0, min(1.0, combined)))

    if probability >= 0.75:
        label = 'Likely AI'
    elif probability <= 0.35:
        label = 'Likely Human'
    else:
        label = 'Unclear'

    breakdown = {
        'gpt2_perplexity': ppx if not math.isnan(ppx) else None,
        'features': features,
        'parts': [{'name': n, 'score': s, 'weight': w} for n, s, w in score_parts],
    }

    return {'probability': probability, 'label': label, 'breakdown': breakdown}


if __name__ == '__main__':
    test = """
    This is a short test text. It may or may not be AI generated. The detector will try to score it.
    """
    print(detect_ai(test, use_gpt2=False))
