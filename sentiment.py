"""
sentiment.py
VADER + TextBlob sentiment (overall and sentence-level) and
entity-level sentiment for the News Bias & Framing Analyzer.
"""

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_VADER = SentimentIntensityAnalyzer()

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def label_from_compound(score: float) -> str:
    """Standard VADER thresholds."""
    if score >= POS_THRESHOLD:
        return "Positive"
    if score <= NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


def vader_scores(text: str) -> dict:
    """Returns neg / neu / pos / compound for a piece of text."""
    return _VADER.polarity_scores(text)


def textblob_scores(text: str) -> dict:
    """Polarity (-1 to +1) and subjectivity (0 = factual, 1 = opinionated)."""
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
    }


def sentence_sentiment(sentences: list) -> pd.DataFrame:
    """Sentiment for every sentence, in article order."""
    rows = []
    for i, sent in enumerate(sentences, start=1):
        v = vader_scores(sent)
        t = textblob_scores(sent)
        rows.append({
            "No.": i,
            "Sentence": sent,
            "VADER": round(v["compound"], 3),
            "TextBlob Polarity": round(t["polarity"], 3),
            "Subjectivity": round(t["subjectivity"], 3),
            "Label": label_from_compound(v["compound"]),
        })
    return pd.DataFrame(rows, columns=[
        "No.", "Sentence", "VADER", "TextBlob Polarity", "Subjectivity", "Label"
    ])


def overall_sentiment(text: str, sent_df: pd.DataFrame) -> dict:
    """Article-level summary built from the whole text and its sentences."""
    v = vader_scores(text)
    t = textblob_scores(text)

    if sent_df.empty:
        return {
            "vader_compound": v["compound"], "vader_pos": v["pos"],
            "vader_neu": v["neu"], "vader_neg": v["neg"],
            "textblob_polarity": t["polarity"],
            "textblob_subjectivity": t["subjectivity"],
            "label": label_from_compound(v["compound"]),
            "avg_sentence_vader": 0.0, "sentiment_volatility": 0.0,
            "pct_positive": 0.0, "pct_negative": 0.0, "pct_neutral": 0.0,
        }

    counts = sent_df["Label"].value_counts(normalize=True) * 100
    return {
        "vader_compound": v["compound"],
        "vader_pos": v["pos"],
        "vader_neu": v["neu"],
        "vader_neg": v["neg"],
        "textblob_polarity": t["polarity"],
        "textblob_subjectivity": t["subjectivity"],
        "label": label_from_compound(v["compound"]),
        "avg_sentence_vader": float(sent_df["VADER"].mean()),
        # std dev of sentence scores: high = emotionally swingy writing
        "sentiment_volatility": float(sent_df["VADER"].std(ddof=0)),
        "pct_positive": float(counts.get("Positive", 0.0)),
        "pct_negative": float(counts.get("Negative", 0.0)),
        "pct_neutral": float(counts.get("Neutral", 0.0)),
    }


def _normalise_entity(name: str) -> str:
    name = name.strip()
    if name.lower().startswith("the "):
        name = name[4:]
    return name.title() if name.islower() else name


def entity_sentiment(entities_df: pd.DataFrame, min_mentions: int = 1) -> pd.DataFrame:
    """
    How is each person/organisation portrayed?
    Method: average VADER score of the sentences in which the entity appears.
    This is an approximation (a sentence can mention several entities),
    but it is a standard, explainable approach for framing analysis.
    """
    cols = ["Entity", "Type", "Mentions", "Avg Sentiment", "Portrayal"]
    if entities_df is None or entities_df.empty:
        return pd.DataFrame(columns=cols)

    df = entities_df.copy()
    df["Entity"] = df["Entity"].apply(_normalise_entity)
    df = df.drop_duplicates(subset=["Entity", "Sentence"])
    df["Score"] = df["Sentence"].apply(lambda s: vader_scores(s)["compound"])

    grouped = (
        df.groupby("Entity")
        .agg(
            Type=("Type", lambda x: x.mode().iat[0]),
            Mentions=("Sentence", "count"),
            AvgSentiment=("Score", "mean"),
        )
        .reset_index()
    )
    grouped["Avg Sentiment"] = grouped["AvgSentiment"].round(3)
    grouped["Portrayal"] = grouped["AvgSentiment"].apply(label_from_compound)
    grouped = grouped[grouped["Mentions"] >= min_mentions]
    return (grouped[cols]
            .sort_values(["Mentions", "Avg Sentiment"], ascending=[False, True])
            .reset_index(drop=True))


if __name__ == "__main__":
    from preprocessing import preprocess

    demo = ("The Prime Minister slammed the opposition on Monday. "
            "Critics say the controversial policy is hurting farmers in Punjab, "
            "while Google announced exciting new investments in Bengaluru. "
            "Experts praised the brilliant plan.")
    out = preprocess(demo)

    sdf = sentence_sentiment(out["sentences"])
    print(sdf[["No.", "VADER", "TextBlob Polarity", "Subjectivity", "Label"]])

    summary = overall_sentiment(out["cleaned"], sdf)
    print("\nOverall:")
    for k, val in summary.items():
        print(f"  {k}: {round(val, 3) if isinstance(val, float) else val}")

    print("\nEntity sentiment:")
    print(entity_sentiment(out["entities"]))