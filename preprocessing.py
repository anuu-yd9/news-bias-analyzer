"""
preprocessing.py
Text cleaning, tokenization, stopword removal, stemming, lemmatization,
POS tagging and named entity recognition for the News Bias & Framing Analyzer.
"""

import re

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))
_NLP = None  # spaCy model is loaded lazily (only when first needed)

ENTITY_LABELS = {"PERSON", "ORG", "GPE", "NORP", "EVENT", "LOC", "FAC"}


def get_nlp():
    """Load the spaCy model once and reuse it."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ---------------------------------------------------------------- cleaning
def clean_text(text: str) -> str:
    """Normalise quotes/whitespace and strip URLs and HTML leftovers."""
    text = text or ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"<[^>]+>", " ", text)                  # HTML tags
    text = (text.replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2014", " - ").replace("\u2013", " - "))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------ tokenization
def split_sentences(text: str) -> list:
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 1]


def tokenize(text: str) -> list:
    """Lowercase word tokens (alphabetic only, apostrophes kept)."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", t)]


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


# ---------------------------------------------- stemming & lemmatization
def stem_tokens(tokens: list) -> list:
    return [_STEMMER.stem(t) for t in tokens]


def _wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_tokens(tokens: list) -> list:
    """WordNet lemmatization using POS tags for accuracy (running -> run)."""
    tagged = nltk.pos_tag(tokens)
    return [_LEMMATIZER.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged]


def stem_vs_lemma_table(tokens: list, limit: int = 25) -> pd.DataFrame:
    """Side-by-side table showing how stemming and lemmatization differ."""
    unique = list(dict.fromkeys(tokens))  # keeps order, removes duplicates
    rows = []
    for tok in unique:
        stem = _STEMMER.stem(tok)
        lemma = lemmatize_tokens([tok])[0]
        if stem != tok or lemma != tok:  # only show words that actually change
            rows.append({"Original": tok, "Stem (Porter)": stem, "Lemma (WordNet)": lemma})
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows, columns=["Original", "Stem (Porter)", "Lemma (WordNet)"])


# ------------------------------------------------------------- POS tagging
def pos_distribution(tokens: list) -> pd.DataFrame:
    """Count of nouns, verbs, adjectives, adverbs in the text."""
    groups = {"Noun": 0, "Verb": 0, "Adjective": 0, "Adverb": 0}
    for _, tag in nltk.pos_tag(tokens):
        if tag.startswith("NN"):
            groups["Noun"] += 1
        elif tag.startswith("VB"):
            groups["Verb"] += 1
        elif tag.startswith("JJ"):
            groups["Adjective"] += 1
        elif tag.startswith("RB"):
            groups["Adverb"] += 1
    return pd.DataFrame({"POS": list(groups.keys()), "Count": list(groups.values())})


# ------------------------------------------------------------------- NER
def extract_entities(text: str) -> pd.DataFrame:
    """Named entities with the sentence each one appears in."""
    doc = get_nlp()(text)
    rows = []
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            rows.append({
                "Entity": ent.text.strip(),
                "Type": ent.label_,
                "Sentence": ent.sent.text.strip(),
            })
    return pd.DataFrame(rows, columns=["Entity", "Type", "Sentence"])


# ----------------------------------------------------------- full pipeline
def preprocess(text: str) -> dict:
    """Run the full pipeline and return every intermediate result."""
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    tokens = tokenize(cleaned)
    no_stop = remove_stopwords(tokens)
    return {
        "cleaned": cleaned,
        "sentences": sentences,
        "tokens": tokens,
        "tokens_no_stop": no_stop,
        "stems": stem_tokens(no_stop),
        "lemmas": lemmatize_tokens(no_stop),
        "stem_lemma_table": stem_vs_lemma_table(no_stop),
        "pos": pos_distribution(tokens),
        "entities": extract_entities(cleaned),
    }


if __name__ == "__main__":
    demo = ("The Prime Minister slammed the opposition on Monday. "
            "Critics say the controversial policy is hurting farmers in Punjab, "
            "while Google announced new investments in Bengaluru.")
    out = preprocess(demo)
    print("Sentences      :", out["sentences"])
    print("Tokens         :", out["tokens"][:12])
    print("No stopwords   :", out["tokens_no_stop"][:12])
    print("Stems          :", out["stems"][:12])
    print("Lemmas         :", out["lemmas"][:12])
    print("\nStem vs Lemma:\n", out["stem_lemma_table"])
    print("\nPOS counts:\n", out["pos"])
    print("\nEntities:\n", out["entities"][["Entity", "Type"]])