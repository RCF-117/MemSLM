"""Runtime evaluation helpers for CLI eval mode.

This module keeps metric/matching logic out of ``main.py`` so the CLI can
focus on orchestration.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List


def normalize_text_for_match(text: str, match_cfg: Dict[str, Any]) -> str:
    """Normalize text using evaluation matching config."""
    normalize_cfg = match_cfg["normalize"]
    value = str(text).strip()
    if bool(normalize_cfg["lowercase"]):
        value = value.lower()
    if bool(normalize_cfg["strip_punctuation"]):
        table = str.maketrans("", "", string.punctuation)
        value = value.translate(table)
    if bool(normalize_cfg["collapse_whitespace"]):
        value = " ".join(value.split())
    if bool(normalize_cfg["strip_articles"]):
        articles = {str(x).strip().lower() for x in normalize_cfg["article_tokens"]}
        value = " ".join(tok for tok in value.split() if tok not in articles)
    return value


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1."""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_count: Dict[str, int] = {}
    for tok in ref_tokens:
        ref_count[tok] = ref_count.get(tok, 0) + 1
    overlap = 0
    for tok in pred_tokens:
        remain = ref_count.get(tok, 0)
        if remain > 0:
            overlap += 1
            ref_count[tok] = remain - 1
    if overlap == 0:
        return 0.0
    precision = overlap / float(len(pred_tokens))
    recall = overlap / float(len(ref_tokens))
    return (2.0 * precision * recall) / (precision + recall)


def extract_numbers(text: str) -> List[float]:
    """Extract numeric values from text."""
    values: List[float] = []
    for item in re.findall(r"[-+]?\d*\.?\d+", str(text)):
        try:
            values.append(float(item))
        except ValueError:
            continue
    return values


def split_expected_answers(expected: str, match_cfg: Dict[str, Any]) -> List[str]:
    """Split expected answer into candidate references."""
    split_cfg = match_cfg["answer_split"]
    raw = str(expected).strip()
    if not raw:
        return []
    if not bool(split_cfg["enabled"]):
        return [raw]

    candidates = [raw]

    # Expand common "also acceptable" patterns into clean alternatives.
    # Example: "14 days. 15 days (including the last day) is also acceptable."
    # -> add "15 days"
    acceptable_patterns = [
        r"([^.?!]+?)\s*\([^)]*\)\s*is also acceptable\.?",
        r"([^.?!]+?)\s+is also acceptable\.?",
    ]
    for pat in acceptable_patterns:
        for m in re.finditer(pat, raw, flags=re.IGNORECASE):
            candidate = str(m.group(1)).strip(" .,:;")
            if candidate:
                candidates.append(candidate)
    for delimiter in split_cfg["delimiters"]:
        next_list: List[str] = []
        for item in candidates:
            pieces = [part.strip() for part in str(item).split(str(delimiter))]
            next_list.extend(pieces)
        candidates = next_list

    for pattern in split_cfg["regex_separators"]:
        next_list = []
        for item in candidates:
            pieces = [part.strip() for part in re.split(str(pattern), str(item))]
            next_list.extend(pieces)
        candidates = next_list

    uniq: List[str] = []
    for cand in candidates:
        cleaned = " ".join(cand.replace("|", " ").split()).strip(" .,:;")
        if cleaned and cleaned not in uniq:
            uniq.append(cleaned)
        if len(uniq) >= int(split_cfg["max_candidates"]):
            break
    return uniq if uniq else [raw]


def evaluate_match(prediction: str, expected: str, eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate prediction against expected answer using configured matching rules."""
    match_cfg = eval_cfg["matching"]
    pred_norm = normalize_text_for_match(prediction, match_cfg)
    references = split_expected_answers(expected, match_cfg)

    best = {
        "is_match": False,
        "best_reference": "",
        "em": 0.0,
        "f1": 0.0,
        "substring": 0.0,
        "numeric": 0.0,
    }
    if not pred_norm or not references:
        return best

    for ref in references:
        ref_norm = normalize_text_for_match(ref, match_cfg)
        if not ref_norm:
            continue
        em = 1.0 if pred_norm == ref_norm else 0.0
        f1 = token_f1(pred_norm, ref_norm)
        substring = (
            1.0
            if bool(match_cfg["substring_match"]) and (ref_norm in pred_norm or pred_norm in ref_norm)
            else 0.0
        )

        numeric = 0.0
        num_cfg = dict(match_cfg.get("numeric_match", {}))
        if bool(num_cfg.get("enabled", False)):
            ref_re = str(num_cfg.get("reference_regex", "")).strip()
            if ref_re and re.match(ref_re, ref_norm):
                pred_nums = extract_numbers(pred_norm)
                ref_nums = extract_numbers(ref_norm)
                if pred_nums and ref_nums:
                    tol = float(match_cfg["numeric_tolerance"])
                    for pn in pred_nums:
                        if any(abs(pn - rn) <= tol for rn in ref_nums):
                            numeric = 1.0
                            break

        mode = str(match_cfg["mode"])
        if mode == "em":
            is_match = em >= 1.0
        elif mode == "f1":
            is_match = f1 >= float(match_cfg["f1_threshold"])
        elif mode == "em_and_f1":
            is_match = em >= 1.0 and f1 >= float(match_cfg["f1_threshold"])
        else:
            is_match = (
                em >= 1.0
                or f1 >= float(match_cfg["f1_threshold"])
                or substring >= 1.0
                or numeric >= 1.0
            )

        if (f1 > best["f1"]) or (f1 == best["f1"] and em > best["em"]):
            best = {
                "is_match": is_match,
                "best_reference": ref,
                "em": em,
                "f1": f1,
                "substring": substring,
                "numeric": numeric,
            }
        elif is_match and not best["is_match"]:
            best["is_match"] = True

    return best


def compute_answer_span_hit(expected: str, chunks: List[Dict[str, Any]], eval_cfg: Dict[str, Any]) -> bool:
    """Strict retrieval hit: expected answer appears in retrieved chunk text."""
    match_cfg = eval_cfg["matching"]
    references = split_expected_answers(expected, match_cfg)
    ref_norms: List[str] = []
    for ref in references:
        ref_norm = normalize_text_for_match(ref, match_cfg)
        if ref_norm:
            ref_norms.append(ref_norm)
    if not ref_norms:
        return False
    for chunk in chunks:
        chunk_text = str(chunk.get("text", "")).strip()
        if not chunk_text:
            continue
        chunk_norm = normalize_text_for_match(chunk_text, match_cfg)
        if not chunk_norm:
            continue
        for ref_norm in ref_norms:
            if ref_norm in chunk_norm:
                return True
    return False


def compute_support_sentence_hit(
    expected: str, chunks: List[Dict[str, Any]], eval_cfg: Dict[str, Any]
) -> bool:
    """Relaxed retrieval hit: enough normalized answer tokens are covered by one chunk."""
    match_cfg = eval_cfg["matching"]
    support_cfg = dict(match_cfg["support_hit"])
    min_ratio = float(support_cfg["min_overlap_ratio"])
    min_tokens = int(support_cfg["min_overlap_tokens"])
    references = split_expected_answers(expected, match_cfg)

    ref_token_sets: List[set[str]] = []
    for ref in references:
        ref_norm = normalize_text_for_match(ref, match_cfg)
        tokens = {tok for tok in ref_norm.split() if tok}
        if tokens:
            ref_token_sets.append(tokens)
    if not ref_token_sets:
        return False

    for chunk in chunks:
        chunk_text = str(chunk.get("text", "")).strip()
        if not chunk_text:
            continue
        chunk_norm = normalize_text_for_match(chunk_text, match_cfg)
        chunk_tokens = {tok for tok in chunk_norm.split() if tok}
        if not chunk_tokens:
            continue
        for ref_tokens in ref_token_sets:
            overlap = len(ref_tokens.intersection(chunk_tokens))
            ratio = float(overlap) / float(len(ref_tokens))
            if overlap >= min_tokens and ratio >= min_ratio:
                return True
    return False


def compute_answer_token_density_from_texts(
    expected: str,
    texts: List[str],
    eval_cfg: Dict[str, Any],
) -> float:
    """Estimate how much of the provided text mass is occupied by answer tokens.

    This is an evaluation-only proxy for prompt noise. A higher value means the
    answer-bearing tokens occupy a larger share of the prompt; a lower value
    means the prompt is dominated by non-answer text.
    """
    match_cfg = eval_cfg["matching"]
    references = split_expected_answers(expected, match_cfg)
    context_tokens: List[str] = []
    for text in texts:
        norm = normalize_text_for_match(text, match_cfg)
        if norm:
            context_tokens.extend([tok for tok in norm.split() if tok])
    if not context_tokens or not references:
        return 0.0

    best_density = 0.0
    for ref in references:
        ref_norm = normalize_text_for_match(ref, match_cfg)
        ref_tokens = {tok for tok in ref_norm.split() if tok}
        if not ref_tokens:
            continue
        overlap = sum(1 for tok in context_tokens if tok in ref_tokens)
        density = float(overlap) / float(len(context_tokens))
        if density > best_density:
            best_density = density
    return max(0.0, min(1.0, best_density))


def compute_noise_density_from_texts(
    expected: str,
    texts: List[str],
    eval_cfg: Dict[str, Any],
) -> float:
    """Complement of answer token density."""
    return max(0.0, 1.0 - compute_answer_token_density_from_texts(expected, texts, eval_cfg))


def compute_answer_token_density(
    expected: str,
    chunks: List[Dict[str, Any]],
    eval_cfg: Dict[str, Any],
) -> float:
    """Estimate answer token density over chunk-like payloads."""
    texts = [str(chunk.get("text", "")).strip() for chunk in chunks if str(chunk.get("text", "")).strip()]
    return compute_answer_token_density_from_texts(expected, texts, eval_cfg)


def compute_noise_density(
    expected: str,
    chunks: List[Dict[str, Any]],
    eval_cfg: Dict[str, Any],
) -> float:
    """Estimate noise density over chunk-like payloads."""
    texts = [str(chunk.get("text", "")).strip() for chunk in chunks if str(chunk.get("text", "")).strip()]
    return compute_noise_density_from_texts(expected, texts, eval_cfg)


def eval_group_key(question_id: str, question_type: str, eval_cfg: Dict[str, Any]) -> str:
    """Build grouped eval key, including abstention split when configured."""
    key = question_type or "unknown"
    if bool(eval_cfg["report_abstention_separately"]) and question_id.endswith("_abs"):
        key = f"{key}__abs"
    return key


def update_group_stats(stats: Dict[str, Dict[str, int]], key: str, is_match: bool) -> None:
    """Update grouped match counters."""
    if key not in stats:
        stats[key] = {"total": 0, "matched": 0}
    stats[key]["total"] += 1
    if is_match:
        stats[key]["matched"] += 1
