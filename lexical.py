"""
lexical.py
Lexical diversity and readability metrics for the News Bias & Framing Analyzer.

Why this matters for bias: very repetitive, low-diversity text often signals
slogan-like or emotionally hammered writing, while high diversity suggests
richer, more varied vocabulary.
"""

import re
from collections import Counter

import pandas as pd


# ------------------------------------------------------------------- MTLD
def _mtld_one_direction(tokens: list, threshold: float = 0.72) -> float:
    """Count how many 'factors' (runs where TTR stays above threshold) occur."""
    factors = 0.0
    types = set()
    count = 0
    for tok in tokens:
        count += 1
        types.add(tok)
        if len(types) / count <= threshold:
            factors += 1
            types = set()
            count = 0
    if count > 0:  # partial factor for the leftover run
        ttr = len(types) / count
        factors += (1 - ttr) / (1 - threshold)
    return len(tokens) / factors if factors > 0 else 0.0


def mtld(tokens: list, threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity: average of a forward and a
    backward pass. Unlike plain TTR it is not distorted by text length.
    Needs roughly 50+ tokens to be reliable.
    """
    if len(tokens) < 10:
        return 0.0
    forward = _mtld_one_direction(tokens, threshold)
    backward = _mtld_one_direction(tokens[::-1], threshold)
    return (forward + backward) / 2


# ------------------------------------------------------------ other metrics
def type_token_ratio(tokens: list) -> float:
    """Unique words / total words (drops as texts get longer)."""
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def root_ttr(tokens: list) -> float:
    """Guiraud's index: types / sqrt(tokens). More stable than plain TTR."""
    return len(set(tokens)) / (len(tokens) ** 0.5) if tokens else 0.0


def hapax_stats(tokens: list) -> dict:
    """Hapax legomena = words that occur exactly once."""
    freq = Counter(tokens)
    hapax = sum(1 for c in freq.values() if c == 1)
    return {
        "hapax_count": hapax,
        "hapax_ratio": hapax / len(tokens) if tokens else 0.0,
    }


def yules_k(tokens: list) -> float:
    """Yule's K: vocabulary repetitiveness. LOWER = more diverse."""
    n = len(tokens)
    if n == 0:
        return 0.0
    freq_of_freq = Counter(Counter(tokens).values())
    m2 = sum((i ** 2) * v for i, v in freq_of_freq.items())
    return 10_000 * (m2 - n) / (n ** 2)


# ------------------------------------------------------------- readability
def _count_syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(tokens: list, sentences: list) -> float:
    """Higher = easier to read (60-70 is plain English, below 30 is very hard)."""
    if not tokens or not sentences:
        return 0.0
    syllables = sum(_count_syllables(t) for t in tokens)
    words_per_sentence = len(tokens) / len(sentences)
    syllables_per_word = syllables / len(tokens)
    return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word


def diversity_label(mtld_value: float) -> str:
    """Rough rule-of-thumb label for presentation purposes only."""
    if mtld_value == 0:
        return "Too short to measure"
    if mtld_value < 50:
        return "Low (repetitive)"
    if mtld_value < 90:
        return "Moderate"
    return "High (varied)"


# ----------------------------------------------------------------- summary
def lexical_metrics(tokens: list, sentences: list) -> dict:
    """All lexical metrics in one dictionary."""
    hapax = hapax_stats(tokens)
    mtld_value = mtld(tokens)
    sentence_lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences] or [0]
    return {
        "total_words": len(tokens),
        "unique_words": len(set(tokens)),
        "ttr": type_token_ratio(tokens),
        "root_ttr": root_ttr(tokens),
        "mtld": mtld_value,
        "diversity_label": diversity_label(mtld_value),
        "hapax_count": hapax["hapax_count"],
        "hapax_ratio": hapax["hapax_ratio"],
        "yules_k": yules_k(tokens),
        "avg_word_length": (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0,
        "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths),
        "flesch_reading_ease": flesch_reading_ease(tokens, sentences),
    }


def metrics_table(metrics: dict) -> pd.DataFrame:
    """Friendly two-column table with a plain-English meaning for each metric."""
    rows = [
        ("Total words", metrics["total_words"], "Length of the article"),
        ("Unique words", metrics["unique_words"], "Distinct vocabulary size"),
        ("Type-Token Ratio", round(metrics["ttr"], 3), "Unique / total words (length-sensitive)"),
        ("Root TTR (Guiraud)", round(metrics["root_ttr"], 3), "Length-corrected TTR"),
        ("MTLD", round(metrics["mtld"], 1), "Length-robust diversity; higher = more varied"),
        ("Diversity level", metrics["diversity_label"], "Rule-of-thumb reading of MTLD"),
        ("Hapax ratio", round(metrics["hapax_ratio"], 3), "Share of words used only once"),
        ("Yule's K", round(metrics["yules_k"], 1), "Repetitiveness; lower = more diverse"),
        ("Avg word length", round(metrics["avg_word_length"], 2), "Longer words = more formal text"),
        ("Avg sentence length", round(metrics["avg_sentence_length"], 1), "Words per sentence"),
        ("Flesch Reading Ease", round(metrics["flesch_reading_ease"], 1), "Higher = easier to read"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Meaning"])


if __name__ == "__main__":
    from preprocessing import preprocess

    demo = (
        "The government announced a sweeping new policy on Monday. Supporters called the "
        "policy bold and necessary, while critics called the policy reckless and dangerous. "
        "Farmers in the north said the policy would hurt their livelihoods, but ministers "
        "insisted the policy would deliver growth. Economists remain divided on whether "
        "the reforms will help or harm ordinary households. The opposition demanded a "
        "debate in parliament, and several analysts warned that the plan lacks detail. "
        "Meanwhile, industry groups welcomed the announcement and urged swift implementation."
    )
    out = preprocess(demo)
    metrics = lexical_metrics(out["tokens"], out["sentences"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(metrics_table(metrics).to_string(index=False))