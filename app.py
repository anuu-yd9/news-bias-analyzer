"""
app.py
Streamlit dashboard for the News Bias & Framing Analyzer.
Run with:  streamlit run app.py
"""

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from bias import CATEGORY_COLORS, analyze_article, distinctive_words
from lexical import metrics_table
from samples import SAMPLES, STORY

st.set_page_config(page_title="News Bias & Framing Analyzer", page_icon="📰", layout="wide")

LABEL_COLORS = {"Positive": "#16a34a", "Neutral": "#94a3b8", "Negative": "#dc2626"}
BIAS_STEPS = [
    {"range": [0, 25], "color": "#bbf7d0"},
    {"range": [25, 50], "color": "#fde68a"},
    {"range": [50, 75], "color": "#fdba74"},
    {"range": [75, 100], "color": "#fca5a5"},
]
SENT_STEPS = [
    {"range": [-1, -0.05], "color": "#fecaca"},
    {"range": [-0.05, 0.05], "color": "#e2e8f0"},
    {"range": [0.05, 1], "color": "#bbf7d0"},
]
CLOUD_SKIP = {"say", "also", "would", "could", "one", "like", "get", "make", "new",
              "may", "take", "go", "come", "see", "tell", "year", "said"}


# ================================================================ helpers
def show(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def gauge(value, title, lo, hi, steps, fmt=".1f"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": fmt},
        title={"text": title},
        gauge={"axis": {"range": [lo, hi]}, "bar": {"color": "#1e293b"}, "steps": steps},
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def fetch_url(url: str):
    """Download and parse an article from a URL (needs internet)."""
    from newspaper import Article
    art = Article(url)
    art.download()
    art.parse()
    return (art.title or url), art.text


@st.cache_data(show_spinner=False)
def cached_analysis(text: str, name: str) -> dict:
    return analyze_article(text, name)


def make_wordcloud(lemmas: list):
    freq = Counter(w for w in lemmas if len(w) > 2 and w not in CLOUD_SKIP)
    if not freq:
        return None
    wc = WordCloud(width=900, height=420, background_color="white",
                   colormap="viridis", max_words=60, random_state=1)
    return wc.generate_from_frequencies(dict(freq)).to_array()


def legend_html() -> str:
    return "".join(
        f'<span style="background:{color};color:#111;padding:2px 8px;border-radius:4px;'
        f'margin-right:6px;font-size:0.85rem">{cat}</span>'
        for cat, color in CATEGORY_COLORS.items()
    )


def highlight_box(html: str) -> str:
    html = html.replace("$", "&#36;")  # stop Streamlit treating $ as maths
    return (f'<div style="background:#ffffff;color:#111;padding:16px;border-radius:8px;'
            f'line-height:1.9">{html}</div>')


def subjectivity_word(value: float) -> str:
    if value < 0.35:
        return "mostly factual"
    if value < 0.5:
        return "a mix of fact and opinion"
    return "highly opinionated"


def summary_row(res: dict) -> dict:
    o, lex, b, c = res["overall"], res["lexical"], res["bias"], res["counts"]
    return {
        "Article": res["name"],
        "Words": lex["total_words"],
        "Bias score": b["score"],
        "Bias level": b["label"],
        "Framing tone": b["tone"],
        "VADER compound": round(o["vader_compound"], 3),
        "Avg sentence VADER": round(o["avg_sentence_vader"], 3),
        "Sentiment volatility": round(o["sentiment_volatility"], 3),
        "TextBlob polarity": round(o["textblob_polarity"], 3),
        "Subjectivity": round(o["textblob_subjectivity"], 3),
        "MTLD": round(lex["mtld"], 1),
        "Type-token ratio": round(lex["ttr"], 3),
        "Hapax ratio": round(lex["hapax_ratio"], 3),
        "Flesch ease": round(lex["flesch_reading_ease"], 1),
        "Negative loaded words": int(c.get("Negative loaded", 0)),
        "Positive loaded words": int(c.get("Positive loaded", 0)),
        "Hedges": int(c.get("Hedge", 0)),
        "Certainty words": int(c.get("Certainty", 0)),
        "Intensifiers": int(c.get("Intensifier", 0)),
        "Stance verbs": int(c.get("Stance verb", 0)),
        "Vague attributions": len(res["vague_phrases"]),
    }


# ============================================================ single view
def render_single(res: dict):
    pre, o, lex, b = res["pre"], res["overall"], res["lexical"], res["bias"]
    counts = res["counts"]
    k = res["name"].replace(" ", "_")

    tabs = st.tabs(["Overview", "Sentiment", "Lexical Diversity", "Bias & Framing",
                    "Keywords", "Entities", "NLP Pipeline", "Export"])

    # ---------------------------------------------------------- overview
    with tabs[0]:
        st.subheader(res["name"])
        c1, c2 = st.columns(2)
        with c1:
            show(gauge(b["score"], "Bias score (0-100)", 0, 100, BIAS_STEPS), f"bias_gauge_{k}")
            st.caption(f"Level: **{b['label']}**  |  Framing tone: **{b['tone']}**")
        with c2:
            show(gauge(o["vader_compound"], "Overall sentiment (VADER)", -1, 1, SENT_STEPS, ".2f"),
                 f"sent_gauge_{k}")
            st.caption(f"Overall label: **{o['label']}**")

        m = st.columns(5)
        m[0].metric("Words", lex["total_words"])
        m[1].metric("Sentences", len(pre["sentences"]))
        m[2].metric("Subjectivity", f"{o['textblob_subjectivity']:.2f}")
        m[3].metric("Lexical diversity (MTLD)", f"{lex['mtld']:.1f}")
        m[4].metric("Reading ease", f"{lex['flesch_reading_ease']:.0f}")

        loaded = counts.get("Negative loaded", 0) + counts.get("Positive loaded", 0)
        st.info(
            f"**{res['name']}** scores **{b['score']} / 100** ({b['label'].lower()} bias indicators). "
            f"Overall sentiment is **{o['label'].lower()}** (VADER {o['vader_compound']:.2f}) and the "
            f"writing is {subjectivity_word(o['textblob_subjectivity'])} "
            f"(subjectivity {o['textblob_subjectivity']:.2f}). It uses **{loaded}** loaded words and "
            f"**{len(res['vague_phrases'])}** vague attributions such as 'critics say'."
        )
        st.caption("The bias score is a heuristic indicator of emotive, one-sided writing style. "
                   "It does not measure whether the facts are true or the article's political leaning.")

    # --------------------------------------------------------- sentiment
    with tabs[1]:
        df = res["sentences_df"]
        st.markdown("#### Sentence-by-sentence sentiment")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["No."], y=df["VADER"], name="VADER",
            marker_color=[LABEL_COLORS[x] for x in df["Label"]],
            customdata=df["Sentence"].str.slice(0, 110),
            hovertemplate="Sentence %{x}<br>VADER %{y}<br>%{customdata}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df["No."], y=df["TextBlob Polarity"], name="TextBlob polarity",
            mode="lines+markers", line=dict(color="#2563eb"),
        ))
        fig.add_hline(y=0, line_color="#64748b", line_width=1)
        fig.update_layout(height=380, xaxis_title="Sentence number",
                          yaxis=dict(range=[-1.05, 1.05], title="Sentiment score"),
                          legend=dict(orientation="h", y=1.12), margin=dict(t=40))
        show(fig, f"timeline_{k}")

        c1, c2 = st.columns([1, 2])
        with c1:
            mix = pd.DataFrame({
                "Label": ["Positive", "Neutral", "Negative"],
                "Share": [o["pct_positive"], o["pct_neutral"], o["pct_negative"]],
            })
            pie = px.pie(mix, names="Label", values="Share", hole=0.5,
                         color="Label", color_discrete_map=LABEL_COLORS)
            pie.update_layout(height=300, margin=dict(t=30, b=10), title="Share of sentences")
            show(pie, f"mix_{k}")
        with c2:
            st.markdown("**Scores**")
            s = st.columns(3)
            s[0].metric("VADER compound", f"{o['vader_compound']:.2f}")
            s[1].metric("Avg sentence VADER", f"{o['avg_sentence_vader']:.2f}")
            s[2].metric("Volatility", f"{o['sentiment_volatility']:.2f}")
            st.caption("Volatility = standard deviation of sentence scores. High = the article swings "
                       "between emotional highs and lows.")
            st.markdown("**Most negative sentences**")
            st.dataframe(df.nsmallest(3, "VADER")[["Sentence", "VADER"]],
                         hide_index=True, use_container_width=True)
            st.markdown("**Most positive sentences**")
            st.dataframe(df.nlargest(3, "VADER")[["Sentence", "VADER"]],
                         hide_index=True, use_container_width=True)

        with st.expander("All sentences"):
            st.dataframe(df, hide_index=True, use_container_width=True)

    # ------------------------------------------------- lexical diversity
    with tabs[2]:
        st.markdown("#### Lexical diversity & readability")
        mt = metrics_table(lex)
        mt["Value"] = mt["Value"].astype(str)
        st.dataframe(mt, hide_index=True, use_container_width=True)
        lengths = [len(s.split()) for s in pre["sentences"]]
        hist = px.histogram(x=lengths, nbins=min(15, max(len(set(lengths)), 1)),
                            labels={"x": "Words per sentence"})
        hist.update_layout(height=300, yaxis_title="Sentences", bargap=0.1,
                           title="Sentence length distribution")
        show(hist, f"hist_{k}")
        with st.expander("How to read these metrics"):
            st.markdown(
                "- **Type-Token Ratio** falls as articles get longer, so compare it only for similar lengths.\n"
                "- **MTLD** is length-robust and is the main diversity measure here (needs 50+ words).\n"
                "- **Hapax ratio** is the share of words used exactly once.\n"
                "- **Yule's K**: lower means a more varied vocabulary.\n"
                "- **Flesch Reading Ease**: higher is easier (60-70 is plain English)."
            )

    # ----------------------------------------------------- bias & framing
    with tabs[3]:
        st.markdown("#### How the bias score is built")
        c1, c2 = st.columns([1, 2])
        with c1:
            show(gauge(b["score"], "Bias score", 0, 100, BIAS_STEPS), f"bias_gauge2_{k}")
        with c2:
            bd = b["breakdown"]
            bar = px.bar(bd, x="Points", y="Component", orientation="h", text="Points")
            bar.update_layout(height=280, margin=dict(t=20, b=10))
            show(bar, f"breakdown_{k}")
        st.dataframe(b["breakdown"], hide_index=True, use_container_width=True)
        st.caption("Each component is scaled between a floor and a cap, multiplied by its weight "
                   "(30/25/15/15/15), and the points are added up. Nothing is hidden.")

        st.markdown("#### Article text with flagged words")
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.markdown(highlight_box(res["highlight"]), unsafe_allow_html=True)

        st.markdown("#### Flagged words")
        ind = res["indicators"]
        if ind.empty:
            st.success("No loaded, hedging, certainty or stance words were detected.")
        else:
            cc = ind.groupby("Category", as_index=False)["Count"].sum()
            cbar = px.bar(cc, x="Category", y="Count", color="Category",
                          color_discrete_map=CATEGORY_COLORS)
            cbar.update_layout(height=300, showlegend=False)
            show(cbar, f"cats_{k}")
            st.dataframe(ind, hide_index=True, use_container_width=True)

        st.markdown("#### Vague attribution")
        if res["vague_phrases"]:
            st.write(", ".join(f"`{p}`" for p in sorted(set(res["vague_phrases"]))))
            st.caption("Phrases that push a view without naming who holds it.")
        else:
            st.success("No vague attribution phrases found.")

    # ---------------------------------------------------------- keywords
    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top keywords (lemmatized)")
            kw = res["keywords"].sort_values("Count")
            kbar = px.bar(kw, x="Count", y="Word", orientation="h")
            kbar.update_layout(height=420, margin=dict(t=20))
            show(kbar, f"kw_{k}")
        with c2:
            st.markdown("#### Word cloud")
            img = make_wordcloud(pre["lemmas"])
            if img is None:
                st.info("Not enough words for a word cloud.")
            else:
                st.image(img)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Repeated two-word phrases")
            if res["bigrams"].empty:
                st.caption("No two-word phrase is repeated in this article.")
            else:
                st.dataframe(res["bigrams"], hide_index=True, use_container_width=True)
        with c4:
            st.markdown("#### Repeated three-word phrases")
            if res["trigrams"].empty:
                st.caption("No three-word phrase is repeated in this article.")
            else:
                st.dataframe(res["trigrams"], hide_index=True, use_container_width=True)

    # ---------------------------------------------------------- entities
    with tabs[5]:
        st.markdown("#### How each person / organisation is portrayed")
        ef = res["entity_framing"]
        if ef.empty:
            st.info("No people, organisations or places were detected in this article.")
        else:
            ebar = px.bar(ef.sort_values("Avg Sentiment"), x="Avg Sentiment", y="Entity",
                          orientation="h", color="Avg Sentiment",
                          color_continuous_scale="RdYlGn", range_color=[-1, 1],
                          hover_data=["Type", "Mentions"])
            ebar.update_layout(height=max(300, 55 * len(ef)), xaxis=dict(range=[-1.05, 1.05]),
                               coloraxis_showscale=False)
            show(ebar, f"ent_{k}")
            st.dataframe(ef, hide_index=True, use_container_width=True)
            st.caption("Each entity is scored on the clause where it appears: VADER score + 0.25 x "
                       "(positive loaded words - negative loaded words), capped at +/-1.")
        with st.expander("Raw named entities (spaCy)"):
            st.dataframe(pre["entities"], hide_index=True, use_container_width=True)

    # ------------------------------------------------------- NLP pipeline
    with tabs[6]:
        st.markdown("#### Every preprocessing step")
        p = st.columns(4)
        p[0].metric("Sentences", len(pre["sentences"]))
        p[1].metric("Tokens", len(pre["tokens"]))
        p[2].metric("After stopword removal", len(pre["tokens_no_stop"]))
        p[3].metric("Unique lemmas", len(set(pre["lemmas"])))
        st.markdown("**1. Cleaned text**")
        with st.expander("Show cleaned text"):
            st.write(pre["cleaned"])
        st.markdown("**2. Tokens (first 40)**")
        st.code(" | ".join(pre["tokens"][:40]))
        st.markdown("**3. After stopword removal (first 40)**")
        st.code(" | ".join(pre["tokens_no_stop"][:40]))
        st.markdown("**4. Stemming (Porter) vs lemmatization (WordNet)**")
        st.dataframe(pre["stem_lemma_table"], hide_index=True, use_container_width=True)
        st.caption("Stems can be non-words (policy -> polici). Lemmas are real dictionary words.")
        st.markdown("**5. Part-of-speech distribution**")
        pbar = px.bar(pre["pos"], x="POS", y="Count", color="POS")
        pbar.update_layout(height=300, showlegend=False)
        show(pbar, f"pos_{k}")

    # ------------------------------------------------------------ export
    with tabs[7]:
        st.markdown("#### Download results (CSV)")
        d = st.columns(4)
        d[0].download_button("Summary", csv_bytes(pd.DataFrame([summary_row(res)])),
                             "summary.csv", "text/csv", key=f"dl_sum_{k}")
        d[1].download_button("Sentence sentiment", csv_bytes(res["sentences_df"]),
                             "sentence_sentiment.csv", "text/csv", key=f"dl_sent_{k}")
        d[2].download_button("Flagged words", csv_bytes(res["indicators"]),
                             "flagged_words.csv", "text/csv", key=f"dl_ind_{k}")
        d[3].download_button("Entity framing", csv_bytes(res["entity_framing"]),
                             "entity_framing.csv", "text/csv", key=f"dl_ent_{k}")


# ======================================================== comparison view
def render_compare(results: list):
    summary = pd.DataFrame([summary_row(r) for r in results])
    top = summary.loc[summary["Bias score"].idxmax()]
    low = summary.loc[summary["Bias score"].idxmin()]
    st.info(
        f"**Highest bias score:** {top['Article']} ({top['Bias score']}, {top['Framing tone'].lower()}).  "
        f"**Lowest:** {low['Article']} ({low['Bias score']}).  "
        f"Gap: **{top['Bias score'] - low['Bias score']:.1f} points** on the same story."
    )

    tabs = st.tabs(["Summary", "Sentiment", "Bias breakdown", "Entity framing",
                    "Word choice", "Highlighted text", "Export"])

    # ----------------------------------------------------------- summary
    with tabs[0]:
        fig = px.bar(summary, x="Article", y="Bias score", color="Bias score", text="Bias score",
                     color_continuous_scale=["#22c55e", "#eab308", "#ef4444"], range_color=[0, 100])
        fig.update_layout(height=380, yaxis=dict(range=[0, 100]), coloraxis_showscale=False,
                          title="Bias score by source")
        show(fig, "cmp_bias")
        melted = summary.melt(id_vars="Article",
                              value_vars=["Avg sentence VADER", "TextBlob polarity", "Subjectivity"])
        gfig = px.bar(melted, x="Article", y="value", color="variable", barmode="group",
                      labels={"value": "Score", "variable": "Metric"})
        gfig.update_layout(height=360, title="Sentiment and subjectivity")
        show(gfig, "cmp_sent_bars")
        st.markdown("#### All metrics side by side")
        st.dataframe(summary.set_index("Article").T.astype(str), use_container_width=True)

    # --------------------------------------------------------- sentiment
    with tabs[1]:
        tl = pd.concat([r["sentences_df"][["No.", "VADER"]].assign(Article=r["name"])
                        for r in results])
        lfig = px.line(tl, x="No.", y="VADER", color="Article", markers=True)
        lfig.add_hline(y=0, line_color="#64748b", line_width=1)
        lfig.update_layout(height=400, xaxis_title="Sentence number",
                           yaxis=dict(range=[-1.05, 1.05], title="VADER score"),
                           title="How sentiment moves through each article")
        show(lfig, "cmp_timeline")
        cols = st.columns(len(results))
        for i, (col, r) in enumerate(zip(cols, results)):
            with col:
                o = r["overall"]
                mix = pd.DataFrame({
                    "Label": ["Positive", "Neutral", "Negative"],
                    "Share": [o["pct_positive"], o["pct_neutral"], o["pct_negative"]],
                })
                pie = px.pie(mix, names="Label", values="Share", hole=0.5, color="Label",
                             color_discrete_map=LABEL_COLORS, title=r["name"])
                pie.update_layout(height=300, margin=dict(t=60, b=10), showlegend=(i == 0))
                show(pie, f"cmp_pie_{i}")

    # ---------------------------------------------------- bias breakdown
    with tabs[2]:
        comp = pd.concat([r["bias"]["breakdown"][["Component", "Points"]].assign(Article=r["name"])
                          for r in results])
        sfig = px.bar(comp, x="Article", y="Points", color="Component", text="Points")
        sfig.update_layout(height=420, yaxis=dict(range=[0, 100], title="Bias score points"),
                           title="Where each score comes from")
        show(sfig, "cmp_stack")
        cat = pd.DataFrame([{"Article": r["name"], "Category": kk, "Count": int(v)}
                            for r in results for kk, v in r["counts"].items()])
        if cat.empty:
            st.success("None of the articles contain flagged words.")
        else:
            cfig = px.bar(cat, x="Category", y="Count", color="Article", barmode="group")
            cfig.update_layout(height=360, title="Flagged words by category")
            show(cfig, "cmp_cats")
        cols = st.columns(len(results))
        for col, r in zip(cols, results):
            with col:
                st.markdown(f"**{r['name']}**")
                if r["indicators"].empty:
                    st.caption("No flagged words.")
                else:
                    st.dataframe(r["indicators"][["Word", "Category", "Count"]],
                                 hide_index=True, use_container_width=True)

    # ---------------------------------------------------- entity framing
    with tabs[3]:
        frames = [r["entity_framing"][["Entity", "Type", "Mentions", "Avg Sentiment", "Portrayal"]]
                  .assign(Article=r["name"]) for r in results if not r["entity_framing"].empty]
        if not frames:
            st.info("No people or organisations were detected.")
        else:
            allent = pd.concat(frames)
            spread = allent.groupby("Entity")["Article"].nunique()
            shared = spread[spread >= 2].index
            if len(shared):
                view = allent[allent["Entity"].isin(shared)]
                st.caption("Showing entities that appear in at least two articles, so you can compare "
                           "how the same person or organisation is framed.")
            else:
                view = allent
                st.caption("No entity appears in more than one article, so all entities are shown.")
            efig = px.bar(view, x="Entity", y="Avg Sentiment", color="Article", barmode="group",
                          range_y=[-1.05, 1.05], hover_data=["Mentions", "Portrayal"])
            efig.update_layout(height=420, title="Portrayal of each entity, by source")
            show(efig, "cmp_ent")
            pivot = (view.pivot_table(index="Entity", columns="Article", values="Avg Sentiment")
                     .round(2).reset_index())
            st.dataframe(pivot, hide_index=True, use_container_width=True)

    # -------------------------------------------------------- word choice
    with tabs[4]:
        if len(results) > 2:
            names = [r["name"] for r in results]
            c1, c2 = st.columns(2)
            na = c1.selectbox("Source A", names, index=0, key="wc_a")
            nb = c2.selectbox("Source B", names, index=1, key="wc_b")
            a = next(r for r in results if r["name"] == na)
            b = next(r for r in results if r["name"] == nb)
        else:
            a, b = results[0], results[1]
        if a["name"] == b["name"]:
            st.warning("Pick two different sources.")
        else:
            only_a, only_b = distinctive_words(a["pre"]["lemmas"], b["pre"]["lemmas"], top=10,
                                               label_a=a["name"], label_b=b["name"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Words {a['name']} leans on**")
                st.dataframe(only_a, hide_index=True, use_container_width=True)
            with c2:
                st.markdown(f"**Words {b['name']} leans on**")
                st.dataframe(only_b, hide_index=True, use_container_width=True)
            both = pd.concat([only_a, only_b]).drop_duplicates("Word")
            if not both.empty:
                wfig = px.bar(both.sort_values("Difference"), x="Difference", y="Word",
                              orientation="h", color="Difference",
                              color_continuous_scale="RdBu", range_color=[-50, 50])
                wfig.update_layout(height=520, coloraxis_showscale=False,
                                   title=f"Positive = more in {a['name']}, negative = more in {b['name']}",
                                   xaxis_title="Difference (uses per 1,000 words)")
                show(wfig, "cmp_words")

    # ---------------------------------------------------- highlighted text
    with tabs[5]:
        st.markdown(legend_html(), unsafe_allow_html=True)
        for i, r in enumerate(results):
            with st.expander(f"{r['name']}  -  bias {r['bias']['score']}", expanded=(i == 0)):
                st.markdown(highlight_box(r["highlight"]), unsafe_allow_html=True)

    # ------------------------------------------------------------ export
    with tabs[6]:
        st.download_button("Download comparison summary (CSV)", csv_bytes(summary),
                           "comparison_summary.csv", "text/csv", key="dl_cmp_sum")
        for i, r in enumerate(results):
            st.download_button(f"Sentence sentiment: {r['name']}", csv_bytes(r["sentences_df"]),
                               f"sentences_{i + 1}.csv", "text/csv", key=f"dl_cmp_sent_{i}")


# =================================================================== inputs
def input_slot(idx: int) -> dict:
    st.markdown(f"**Article {idx + 1}**")
    source = st.radio("Source", ["Sample", "Paste text", "URL"], key=f"src{idx}",
                      horizontal=True, label_visibility="collapsed")
    if source == "Sample":
        keys = list(SAMPLES.keys())
        choice = st.selectbox("Sample article", keys, index=idx % len(keys), key=f"sample{idx}")
        st.caption(f"Story: {STORY}. Fictional text for demonstration.")
        return {"name": choice, "text": SAMPLES[choice].strip(), "url": None}
    if source == "Paste text":
        name = st.text_input("Source name", value=f"Pasted article {idx + 1}", key=f"label{idx}")
        text = st.text_area("Article text", height=230, key=f"text{idx}",
                            placeholder="Paste the full article text here (at least 30 words)...")
        return {"name": name.strip() or f"Pasted article {idx + 1}", "text": text, "url": None}
    url = st.text_input("Article URL", key=f"url{idx}", placeholder="https://...")
    return {"name": url, "text": "", "url": url.strip()}


def uniquify(names: list) -> list:
    seen, out = Counter(), []
    for n in names:
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


# ===================================================================== main
def main():
    st.title("News Bias & Framing Analyzer")
    st.caption("Sentiment, lexical diversity, framing and bias indicators for news articles "
               "(Web & Social Media Analytics)")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Single article", "Compare articles"])
        n = 1
        if mode == "Compare articles":
            n = st.slider("Number of articles", 2, 3, 3)
        st.divider()
        st.markdown("### Techniques used")
        st.markdown(
            "- Cleaning, sentence and word tokenization\n"
            "- Stopword removal\n"
            "- Stemming (Porter) vs lemmatization (WordNet)\n"
            "- POS tagging and named entity recognition\n"
            "- Sentiment: VADER + TextBlob\n"
            "- Lexical diversity: TTR, MTLD, hapax, Yule's K\n"
            "- Loaded-word, hedge and certainty lexicons\n"
            "- N-grams and entity-level framing"
        )
        with st.expander("How the bias score works"):
            st.markdown(
                "Score out of 100 from five visible parts:\n\n"
                "1. Loaded language (30)\n2. Subjectivity (25)\n3. Emotional intensity (15)\n"
                "4. Certainty and intensifiers (15)\n5. Vague attribution and stance verbs (15)\n\n"
                "Bands: under 25 low, 25-50 moderate, 50-75 high, 75+ very high."
            )
        st.caption("A heuristic for emotive writing style, not a fact-check or a political label.")

    slots = []
    cols = st.columns(n)
    for i, col in enumerate(cols):
        with col:
            slots.append(input_slot(i))

    run = st.button("Analyze", type="primary")
    state_key = f"{mode}-{n}"

    if run:
        pairs, problems = [], []
        with st.spinner("Running the NLP pipeline..."):
            for i, slot in enumerate(slots, start=1):
                name, text = slot["name"], slot["text"]
                if slot["url"]:
                    try:
                        name, text = fetch_url(slot["url"])
                        name = name[:60]
                    except Exception as exc:
                        problems.append(f"Article {i}: could not fetch the URL ({exc}). "
                                        "Try pasting the text instead.")
                        continue
                if len(text.split()) < 30:
                    problems.append(f"Article {i}: needs at least 30 words of text.")
                    continue
                pairs.append((name, text))

            names = uniquify([p[0] for p in pairs])
            results = [cached_analysis(text, nm) for nm, (_, text) in zip(names, pairs)]

        for msg in problems:
            st.warning(msg)
        if mode == "Compare articles" and len(results) < 2:
            st.error("Comparison needs at least two valid articles.")
            results = []
        st.session_state["results"] = results or None
        st.session_state["results_key"] = state_key

    results = st.session_state.get("results")
    if results and st.session_state.get("results_key") == state_key:
        st.divider()
        if len(results) == 1:
            render_single(results[0])
        else:
            render_compare(results)
    else:
        st.info("Choose your article(s) above and click **Analyze**.")


main()