"""
bias.py
Bias indicators, framing analysis and the transparent 0-100 bias score.

The score is a HEURISTIC INDICATOR of biased/emotive writing style, not a
verdict on truth or political leaning. Every component is shown on screen.
"""

import re
from collections import Counter
from html import escape

import pandas as pd
from nltk.corpus import stopwords

from lexical import lexical_metrics
from preprocessing import lemmatize_tokens, preprocess
from sentiment import label_from_compound, overall_sentiment, sentence_sentiment, vader_scores

_STOPWORDS = set(stopwords.words("english"))

# --------------------------------------------------------------- lexicons
NEGATIVE_LOADED = {
    "slam", "blast", "lash", "assail", "attack", "rant", "tirade", "smear", "scandal",
    "fiasco", "debacle", "disaster", "catastrophe", "catastrophic", "chaos", "chaotic",
    "crisis", "reckless", "dangerous", "radical", "extremist", "extreme", "corrupt",
    "corruption", "regime", "crackdown", "outrage", "outrageous", "shocking", "shock",
    "disgraceful", "shameful", "brutal", "ruthless", "devastating", "devastate", "plunge",
    "plummet", "collapse", "bungle", "blunder", "sham", "propaganda", "puppet", "thug",
    "mob", "hysteria", "meltdown", "backlash", "fury", "furious", "slump", "toxic",
    "sinister", "draconian", "bizarre", "absurd", "ridiculous", "failure", "betray",
    "betrayal", "liar", "fraud", "cronyism", "incompetent", "incompetence",
    "controversial", "divisive", "embattled", "beleaguered", "panic", "menace",
}
POSITIVE_LOADED = {
    "landmark", "historic", "brilliant", "bold", "heroic", "hero", "triumph", "triumphant",
    "stellar", "visionary", "groundbreaking", "remarkable", "stunning", "flawless",
    "miraculous", "praise", "hail", "celebrate", "boost", "victory", "breakthrough",
    "exciting", "impressive", "masterful", "courageous", "champion", "landslide",
    "thrive", "flourish", "booming", "soar", "spectacular", "fantastic", "tremendous",
    "superb", "genius", "inspiring", "unprecedented",
}
INTENSIFIERS = {
    "very", "extremely", "highly", "incredibly", "absolutely", "totally", "completely",
    "utterly", "deeply", "hugely", "remarkably", "immensely", "terribly", "awfully",
    "really", "truly", "massively", "enormously",
}
CERTAINTY = {
    "clearly", "obviously", "undoubtedly", "certainly", "definitely", "undeniably",
    "unquestionably", "always", "never", "everyone", "nobody", "prove", "proven",
    "inevitable", "inevitably", "indisputable", "surely", "plainly",
}
HEDGES = {
    "may", "might", "could", "possibly", "perhaps", "reportedly", "allegedly",
    "apparently", "seemingly", "likely", "unlikely", "probably", "suggest", "appear",
    "seem", "presumably", "supposedly", "potentially", "arguably", "maybe", "somewhat",
    "roughly", "approximately",
}
STANCE_VERBS = {
    "claim", "admit", "insist", "allege", "boast", "concede", "confess", "whine",
    "complain", "lament", "blame", "accuse", "denounce", "dismiss",
}

# order matters: a word is counted under the first category it matches
LEXICON = {
    "Negative loaded": NEGATIVE_LOADED,
    "Positive loaded": POSITIVE_LOADED,
    "Intensifier": INTENSIFIERS,
    "Certainty": CERTAINTY,
    "Hedge": HEDGES,
    "Stance verb": STANCE_VERBS,
}

CATEGORY_COLORS = {
    "Negative loaded": "#fecaca",
    "Positive loaded": "#bbf7d0",
    "Intensifier": "#fde68a",
    "Certainty": "#ddd6fe",
    "Hedge": "#bae6fd",
    "Stance verb": "#fbcfe8",
}

VAGUE_ATTRIBUTION = [
    r"\bcritics (?:say|argue|claim|charge|warn)\b",
    r"\bexperts (?:say|believe|warn|fear|argue)\b",
    r"\bsome (?:say|believe|argue|claim)\b",
    r"\bmany (?:say|believe|argue|feel)\b",
    r"\bpeople (?:say|believe)\b",
    r"\b(?:sources|insiders|observers|officials) (?:say|said|suggest|claim|believe)\b",
    r"\bit is (?:clear|obvious|undeniable|widely believed)\b",
    r"\bwidely (?:believed|seen|regarded|considered)\b",
    r"\baccording to (?:some|critics|observers)\b",
]

# --------------------------------------------------------- indicator search
def find_indicators(tokens: list) -> pd.DataFrame:
    """Find every loaded/hedge/certainty/intensifier/stance word (matches on word or lemma)."""
    cols = ["Word", "Lemma", "Category", "Count"]
    lemmas = lemmatize_tokens(tokens)
    counter = Counter()
    for tok, lem in zip(tokens, lemmas):
        for category, words in LEXICON.items():
            if tok in words or lem in words:
                counter[(tok, lem, category)] += 1
                break
    rows = [{"Word": w, "Lemma": l, "Category": c, "Count": n}
            for (w, l, c), n in counter.items()]
    if not rows:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(rows, columns=cols)
            .sort_values(["Count", "Word"], ascending=[False, True])
            .reset_index(drop=True))


def category_counts(indicators: pd.DataFrame) -> dict:
    if indicators.empty:
        return {}
    return indicators.groupby("Category")["Count"].sum().to_dict()


def vague_attribution(cleaned_text: str) -> list:
    """Phrases like 'critics say' that push a view without naming a source."""
    text = cleaned_text.lower()
    found = []
    for pattern in VAGUE_ATTRIBUTION:
        found.extend(m.group(0) for m in re.finditer(pattern, text))
    return found


def framing_tone(counts: dict) -> str:
    neg = counts.get("Negative loaded", 0)
    pos = counts.get("Positive loaded", 0)
    if neg + pos == 0:
        return "No loaded language"
    if neg >= 2 * pos:
        return "Negative-leaning"
    if pos >= 2 * neg:
        return "Positive-leaning"
    return "Mixed"


# --------------------------------------------------------------- bias score
# (name, what is measured, floor, cap, weight). Values at/below floor score 0,
# at/above cap score full weight, in between it is linear. Weights sum to 100.
COMPONENTS = [
    ("Loaded language", "loaded words per 100 words", 0.0, 6.0, 30),
    ("Subjectivity", "TextBlob subjectivity (0-1)", 0.20, 0.60, 25),
    ("Emotional intensity", "average |VADER| per sentence", 0.0, 0.50, 15),
    ("Certainty & intensifiers", "per 100 words", 0.0, 4.0, 15),
    ("Vague attribution & stance verbs", "per 100 words", 0.0, 3.0, 15),
]


def _scale(value: float, floor: float, cap: float) -> float:
    if cap <= floor:
        return 0.0
    return min(max((value - floor) / (cap - floor), 0.0), 1.0)


def bias_label(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Very high"


def compute_bias_score(counts, total_words, overall, sent_df, vague_count) -> dict:
    total = max(total_words, 1)
    loaded = counts.get("Negative loaded", 0) + counts.get("Positive loaded", 0)
    raw = {
        "Loaded language": loaded / total * 100,
        "Subjectivity": overall["textblob_subjectivity"],
        "Emotional intensity": float(sent_df["VADER"].abs().mean()) if not sent_df.empty else 0.0,
        "Certainty & intensifiers":
            (counts.get("Intensifier", 0) + counts.get("Certainty", 0)) / total * 100,
        "Vague attribution & stance verbs":
            (vague_count + counts.get("Stance verb", 0)) / total * 100,
    }
    rows, score = [], 0.0
    for name, measure, floor, cap, weight in COMPONENTS:
        scaled = _scale(raw[name], floor, cap)
        points = scaled * weight
        score += points
        rows.append({
            "Component": name,
            "Measure": measure,
            "Raw value": round(raw[name], 2),
            "Floor - Cap": f"{floor} - {cap}",
            "Scaled (0-1)": round(scaled, 2),
            "Weight": weight,
            "Points": round(points, 1),
        })
    score = round(score, 1)
    return {
        "score": score,
        "label": bias_label(score),
        "tone": framing_tone(counts),
        "breakdown": pd.DataFrame(rows),
    }


# ----------------------------------------------------------- framing tools
_GENERIC = {"say", "also", "would", "could", "one", "like", "get", "make", "year",
            "new", "may", "take", "go", "come", "see", "tell"}


def top_keywords(lemmas: list, top: int = 15) -> pd.DataFrame:
    counts = Counter(w for w in lemmas if w not in _GENERIC)
    return pd.DataFrame(counts.most_common(top), columns=["Word", "Count"])


def top_ngrams(tokens: list, n: int = 2, top: int = 10, min_count: int = 2) -> pd.DataFrame:
    """Most repeated phrases (these show which words the writer keeps pairing)."""
    grams = []
    for i in range(len(tokens) - n + 1):
        gram = tokens[i:i + n]
        if gram[0] in _STOPWORDS or gram[-1] in _STOPWORDS:
            continue
        if any(len(w) < 3 for w in gram):
            continue
        grams.append(" ".join(gram))
    rows = [(g, c) for g, c in Counter(grams).most_common() if c >= min_count][:top]
    return pd.DataFrame(rows, columns=["Phrase", "Count"])


_CLAUSE_SPLIT = re.compile(
    r",\s*(?:while|but|whereas|although|though)\s+|;\s*|"
    r"\s+(?:while|but|whereas|although|though)\s+|\s+-\s+",
    re.IGNORECASE,
)


def _clean_entity(name: str) -> str:
    name = name.strip()
    if name.lower().startswith("the "):
        name = name[4:]
    if name.endswith("'s"):
        name = name[:-2]
    return name.title() if name.islower() else name


def _clause_for(entity: str, sentence: str) -> str:
    """The part of the sentence that actually contains the entity."""
    for clause in _CLAUSE_SPLIT.split(sentence):
        if entity.lower() in clause.lower():
            return clause
    return sentence


ENTITY_TYPES = {"PERSON", "ORG", "GPE", "NORP", "EVENT"}


def _loaded_in_clause(clause: str) -> list:
    """List of (word, category) for every loaded word inside a clause."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", clause.lower())
    ind = find_indicators(words)
    if ind.empty:
        return []
    ind = ind[ind["Category"].isin(["Negative loaded", "Positive loaded"])]
    found = []
    for row in ind.itertuples():
        found.extend([(row.Word, row.Category)] * int(row.Count))
    return found


def _clause_score(clause: str, loaded: list) -> float:
    """VADER score nudged by loaded words (VADER's dictionary misses many of them)."""
    base = vader_scores(clause)["compound"]
    pos = sum(1 for _, cat in loaded if cat == "Positive loaded")
    neg = sum(1 for _, cat in loaded if cat == "Negative loaded")
    return max(-1.0, min(1.0, base + 0.25 * (pos - neg)))


def _merge_short_names(df: pd.DataFrame) -> pd.Series:
    """Map 'Mehta' -> 'Rohan Mehta' when exactly one longer name ends with it."""
    full = {n for n in df["Name"] if len(n.split()) > 1}
    mapping = {}
    for n in df["Name"].unique():
        if len(n.split()) == 1:
            matches = [f for f in full if f.split()[-1].lower() == n.lower()]
            if len(matches) == 1:
                mapping[n] = matches[0]
    return df["Name"].replace(mapping)


def entity_framing(entities_df: pd.DataFrame) -> pd.DataFrame:
    """
    How is each person/organisation portrayed? Scores only the CLAUSE where the
    entity appears, nudged by loaded words, and lists those words.
    """
    cols = ["Entity", "Type", "Mentions", "Avg Sentiment", "Portrayal", "Loaded words nearby"]
    if entities_df is None or entities_df.empty:
        return pd.DataFrame(columns=cols)

    df = entities_df[entities_df["Type"].isin(ENTITY_TYPES)]
    df = df.drop_duplicates(subset=["Entity", "Sentence"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["Name"] = df["Entity"].apply(_clean_entity)
    df["Name"] = _merge_short_names(df)
    df["Clause"] = df.apply(lambda r: _clause_for(r["Entity"], r["Sentence"]), axis=1)
    df = df.drop_duplicates(subset=["Name", "Clause"])
    df["Loaded"] = df["Clause"].apply(_loaded_in_clause)
    df["Score"] = df.apply(lambda r: _clause_score(r["Clause"], r["Loaded"]), axis=1)

    grouped = df.groupby("Name").agg(
        Type=("Type", lambda x: x.mode().iat[0]),
        Mentions=("Clause", "count"),
        Avg=("Score", "mean"),
        Loaded=("Loaded", lambda x: ", ".join(sorted({w for lst in x for w, _ in lst}))),
    ).reset_index()

    grouped["Avg Sentiment"] = grouped["Avg"].round(3)
    grouped["Portrayal"] = grouped["Avg"].apply(label_from_compound)
    grouped = grouped.rename(columns={"Name": "Entity", "Loaded": "Loaded words nearby"})
    return (grouped[cols]
            .sort_values(["Mentions", "Avg Sentiment"], ascending=[False, True])
            .reset_index(drop=True))
def distinctive_words(lemmas_a: list, lemmas_b: list, top: int = 10,
                      label_a: str = "A", label_b: str = "B"):
    """Words one source uses much more than the other (per 1,000 words)."""
    ca, cb = Counter(lemmas_a), Counter(lemmas_b)
    na, nb = max(len(lemmas_a), 1), max(len(lemmas_b), 1)
    rows = []
    for w in set(ca) | set(cb):
        if w in _GENERIC:
            continue
        pa, pb = ca[w] / na * 1000, cb[w] / nb * 1000
        rows.append({"Word": w, f"{label_a} per 1k": round(pa, 1),
                     f"{label_b} per 1k": round(pb, 1), "Difference": round(pa - pb, 1)})
    if not rows:
        empty = pd.DataFrame(columns=["Word", f"{label_a} per 1k", f"{label_b} per 1k", "Difference"])
        return empty, empty
    df = pd.DataFrame(rows)
    only_a = df.sort_values("Difference", ascending=False).head(top).reset_index(drop=True)
    only_b = df.sort_values("Difference").head(top).reset_index(drop=True)
    return only_a, only_b


def highlight_html(text: str, indicators: pd.DataFrame) -> str:
    """Article text as HTML with indicator words colour-highlighted."""
    lookup = {}
    if not indicators.empty:
        lookup = {row.Word: row.Category for row in indicators.itertuples()}

    def repl(match):
        word = match.group(0)
        category = lookup.get(word.lower())
        if not category:
            return word
        return (f'<mark style="background:{CATEGORY_COLORS[category]};color:#111;'
                f'padding:0 3px;border-radius:3px" title="{category}">{word}</mark>')

    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", repl, escape(text, quote=False))


# ------------------------------------------------------------ full analysis
def analyze_article(text: str, name: str = "Article") -> dict:
    """Run every analysis once and return everything the dashboard needs."""
    pre = preprocess(text)
    sent_df = sentence_sentiment(pre["sentences"])
    overall = overall_sentiment(pre["cleaned"], sent_df)
    lex = lexical_metrics(pre["tokens"], pre["sentences"])
    indicators = find_indicators(pre["tokens"])
    counts = category_counts(indicators)
    vague = vague_attribution(pre["cleaned"])
    bias = compute_bias_score(counts, len(pre["tokens"]), overall, sent_df, len(vague))
    return {
        "name": name,
        "pre": pre,
        "sentences_df": sent_df,
        "overall": overall,
        "lexical": lex,
        "indicators": indicators,
        "counts": counts,
        "vague_phrases": vague,
        "bias": bias,
        "keywords": top_keywords(pre["lemmas"]),
        "bigrams": top_ngrams(pre["tokens"], 2),
        "trigrams": top_ngrams(pre["tokens"], 3),
        "entity_framing": entity_framing(pre["entities"]),
        "highlight": highlight_html(pre["cleaned"], indicators),
    }


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 50)

    loaded_text = (
        "The reckless government slammed farmers with a shocking new law on Monday. "
        "Critics say the controversial policy is a disaster that will devastate rural families. "
        "Ministers insisted the plan is absolutely essential, but experts say it is clearly "
        "a dangerous blunder. The opposition blasted the regime for its corrupt handling of "
        "the crisis. Supporters, however, hailed the bold reform as a historic breakthrough."
    )
    neutral_text = (
        "The government announced a new farm law on Monday. The policy changes how crops are "
        "purchased and sets minimum prices for wheat and rice. Officials said the measure will "
        "take effect in January. Farmer unions and industry groups have asked for a review, and "
        "a parliamentary committee will examine the proposal next month. Economists said the "
        "effects on income may become clear after the first harvest."
    )

    a = analyze_article(loaded_text, "Loaded")
    b = analyze_article(neutral_text, "Neutral")

    for res in (a, b):
        print("=" * 70)
        print(f"{res['name']}: bias score {res['bias']['score']} "
              f"({res['bias']['label']}), tone: {res['bias']['tone']}")
        print(res["bias"]["breakdown"][["Component", "Raw value", "Weight", "Points"]]
              .to_string(index=False))

    print("\nIndicator words in the LOADED article:")
    print(a["indicators"].to_string(index=False))
    print("\nVague attribution found:", a["vague_phrases"])
    print("\nEntity framing:")
    print(a["entity_framing"])

    only_a, only_b = distinctive_words(a["pre"]["lemmas"], b["pre"]["lemmas"],
                                       top=5, label_a="Loaded", label_b="Neutral")
    print("\nWords the LOADED article leans on:\n", only_a)
    print("\nWords the NEUTRAL article leans on:\n", only_b)