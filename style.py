"""
style.py
Newspaper / vintage broadsheet theme for the Streamlit dashboard:
CSS, masthead, hero section, stat cards and the Plotly chart template.
"""

from datetime import date
from html import escape

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK = "#1a1a1a"
RED = "#8b1a1a"
GREEN = "#3f6b4a"
OCHRE = "#b8892b"
BLUE = "#2f4b66"
GREY = "#a39b8b"

REPO_URL = "https://github.com/anuu-yd9/news-bias-analyzer"

# muted "highlighter pen" colours for flagged words
HIGHLIGHTS = {
    "Negative loaded": "#f0b8b0",
    "Positive loaded": "#c5dcb8",
    "Intensifier": "#f2dd9b",
    "Certainty": "#cdc3e3",
    "Hedge": "#b9d3e0",
    "Stance verb": "#e8c3d4",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;0,900;1,700;1,900&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=UnifrakturCook:wght@700&display=swap');

:root{--ink:#1a1a1a;--paper:#f4efe4;--card:#faf6ec;--muted:#5a5446;--rule:#d9d0bb;--red:#8b1a1a;}

.stApp{background-color:var(--paper);background-image:radial-gradient(rgba(26,26,26,.05) 1px,transparent 1px);background-size:5px 5px;color:var(--ink);font-family:'Source Serif 4',Georgia,serif;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{visibility:hidden;height:0;}
[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"]{display:none;}
.block-container{max-width:1180px;padding-top:1.2rem;padding-bottom:3rem;}

h1,h2,h3,h4,h5,h6{font-family:'Playfair Display',Georgia,serif !important;color:var(--ink) !important;letter-spacing:-0.01em;}
h4{border-bottom:4px double var(--ink);padding-bottom:.3rem !important;margin-top:1.4rem !important;font-size:1.35rem !important;}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{font-family:'Source Serif 4',Georgia,serif;font-size:1.02rem;line-height:1.65;}
[data-testid="stCaptionContainer"]{font-style:italic;color:var(--muted);}
hr{border:0 !important;border-top:1px solid var(--ink) !important;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:2px solid var(--ink);background:transparent;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:0;padding:.7rem 1.15rem;height:auto;}
.stTabs [data-baseweb="tab"] p{font-family:'IBM Plex Mono',monospace !important;font-size:.76rem !important;letter-spacing:.12em;text-transform:uppercase;margin:0;color:var(--muted);}
.stTabs [aria-selected="true"]{background:var(--ink) !important;}
.stTabs [aria-selected="true"] p{color:var(--paper) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--red);height:3px;}
.stTabs [data-baseweb="tab-border"]{background:transparent;}

/* buttons */
.stButton > button,[data-testid="stBaseButton-primary"]{font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.14em;border-radius:0;border:2px solid var(--ink);background:var(--ink);color:var(--paper);padding:.6rem 1.8rem;}
.stButton > button p,[data-testid="stBaseButton-primary"] p{color:var(--paper) !important;font-family:'IBM Plex Mono',monospace !important;font-size:.82rem !important;}
.stButton > button:hover,[data-testid="stBaseButton-primary"]:hover{background:var(--red);border-color:var(--red);}
[data-testid="stDownloadButton"] button{background:transparent;border:1.5px solid var(--ink);border-radius:0;color:var(--ink);}
[data-testid="stDownloadButton"] button p{color:var(--ink) !important;font-family:'IBM Plex Mono',monospace !important;font-size:.78rem !important;letter-spacing:.06em;}
[data-testid="stDownloadButton"] button:hover{background:var(--ink);}
[data-testid="stDownloadButton"] button:hover p{color:var(--paper) !important;}

/* inputs */
[data-baseweb="base-input"],[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"] > div{background:var(--card) !important;border-radius:0 !important;}
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"] > div{border:1px solid var(--ink) !important;}
input,textarea{background:var(--card) !important;color:var(--ink) !important;}
[data-testid="stRadio"] label p{font-family:'IBM Plex Mono',monospace !important;font-size:.78rem !important;text-transform:uppercase;letter-spacing:.08em;}

/* metrics, charts, tables, alerts */
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--ink);border-top:4px solid var(--ink);padding:.9rem 1rem;}
[data-testid="stMetricLabel"] p{font-family:'IBM Plex Mono',monospace !important;font-size:.68rem !important;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);}
[data-testid="stMetricValue"]{font-family:'Playfair Display',Georgia,serif;font-weight:700;}
[data-testid="stPlotlyChart"]{background:var(--card);border:1px solid var(--ink);padding:.4rem;}
[data-testid="stDataFrame"]{border:1px solid var(--ink);}
[data-testid="stImage"] img{border:1px solid var(--ink);}
[data-testid="stAlert"],[data-testid="stAlert"] > div{background:var(--card) !important;color:var(--ink) !important;border-radius:0 !important;}
[data-testid="stAlert"]{border:1px solid var(--ink);border-left:6px solid var(--red);}
[data-testid="stAlert"] p{color:var(--ink) !important;}
[data-testid="stExpander"]{border:1px solid var(--ink);border-radius:0;background:var(--card);}
[data-testid="stExpander"] summary p{font-family:'IBM Plex Mono',monospace !important;font-size:.78rem !important;text-transform:uppercase;letter-spacing:.08em;}

/* masthead */
.mast{margin-bottom:.6rem;}
.mast-bar{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;padding:.45rem 0;border-top:4px solid var(--ink);border-bottom:1px solid var(--ink);}
.mast-bar a{color:var(--ink);text-decoration:underline;}
.mast-title{font-family:'UnifrakturCook','Playfair Display',serif;font-size:clamp(2.6rem,7vw,4.6rem);text-align:center;line-height:1.1;padding:.5rem 0 .3rem;}
.mast-sub{text-align:center;font-family:'IBM Plex Mono',monospace;font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;padding:.5rem 0;border-top:1px solid var(--ink);border-bottom:4px double var(--ink);}

/* hero */
.hero{display:grid;grid-template-columns:1.25fr 1fr;gap:2.5rem;align-items:center;padding:2rem 0 1.2rem;}
.kicker{font-family:'IBM Plex Mono',monospace;font-size:.74rem;letter-spacing:.2em;text-transform:uppercase;color:var(--red);font-weight:500;}
.hero-h{font-family:'Playfair Display',Georgia,serif;font-weight:900;font-size:clamp(2.4rem,5.2vw,4.1rem);line-height:1.04;margin:.6rem 0 1rem;color:var(--ink);}
.hero-h em{color:var(--red);}
.hero-p{font-size:1.08rem;line-height:1.7;color:#3b372f;max-width:34rem;}
.pills{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.2rem;}
.pill{border:1px solid var(--ink);padding:.25rem .8rem;font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;background:var(--card);}
.clip{background:var(--card);border:1px solid var(--ink);box-shadow:7px 7px 0 var(--ink);padding:1.5rem 1.8rem;transform:rotate(.5deg);}
.clip-lab{font-family:'IBM Plex Mono',monospace;font-size:.68rem;letter-spacing:.2em;text-transform:uppercase;color:var(--red);border-bottom:1px solid var(--ink);padding-bottom:.5rem;margin-bottom:.9rem;}
.clip-txt{font-family:'Playfair Display',Georgia,serif;font-style:italic;font-size:1.35rem;line-height:1.75;}
.clip-foot{margin-top:1rem;padding-top:.7rem;border-top:1px dashed var(--ink);font-size:.88rem;color:var(--muted);}

/* cards */
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0 2rem;}
.stat,.card{background:var(--card);border:1px solid var(--ink);border-top:5px solid var(--ink);padding:1.1rem 1.3rem;height:100%;}
.stat .lab,.card .lab{font-family:'IBM Plex Mono',monospace;font-size:.68rem;letter-spacing:.2em;text-transform:uppercase;color:var(--red);}
.stat .num{font-family:'Playfair Display',Georgia,serif;font-weight:900;font-size:2.6rem;line-height:1.15;}
.card .ctitle{font-family:'Playfair Display',Georgia,serif;font-weight:700;font-size:1.3rem;margin:.3rem 0;}
.stat .txt,.card .txt{color:var(--muted);font-size:.95rem;line-height:1.5;}

/* sections, callouts, misc */
.sec{margin:2rem 0 1rem;}
.sec-t{font-family:'Playfair Display',Georgia,serif;font-weight:700;font-size:1.9rem;border-bottom:4px double var(--ink);padding-bottom:.35rem;}
.callout{background:var(--card);border:1px solid var(--ink);border-left:6px solid var(--red);padding:1rem 1.3rem;margin:1rem 0;font-size:1.02rem;line-height:1.65;}
.callout .tag{display:block;font-family:'IBM Plex Mono',monospace;font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;color:var(--red);margin-bottom:.3rem;}
.slot-label{font-family:'IBM Plex Mono',monospace;font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;border-bottom:1px solid var(--ink);padding-bottom:.3rem;margin:.6rem 0 .4rem;}
.foot{margin-top:3rem;padding-top:.8rem;border-top:4px double var(--ink);text-align:center;font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);line-height:1.9;}

@media (max-width:900px){.hero{grid-template-columns:1fr;}.stats{grid-template-columns:1fr;}.mast-bar span:nth-child(2){display:none;}}
"""


def _flat(html: str) -> str:
    """Collapse indentation so Streamlit's markdown never treats HTML as a code block."""
    return " ".join(line.strip() for line in html.splitlines() if line.strip())


def inject_css():
    st.markdown(f"<style>{_flat(_CSS)}</style>", unsafe_allow_html=True)


def register_plotly_theme():
    """Newsprint look for every Plotly chart."""
    body = "Source Serif 4, Georgia, serif"
    pio.templates["newsprint"] = go.layout.Template(
        layout=dict(
            font=dict(family=body, color=INK, size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[INK, RED, OCHRE, BLUE, GREEN, GREY],
            xaxis=dict(gridcolor="#d9d0bb", linecolor=INK, zerolinecolor=INK),
            yaxis=dict(gridcolor="#d9d0bb", linecolor=INK, zerolinecolor=INK),
            title=dict(font=dict(family="Playfair Display, Georgia, serif", size=18)),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor=INK, font=dict(color="#f4efe4", family=body)),
        )
    )
    pio.templates.default = "newsprint"


def _mark(word: str, category: str) -> str:
    return (f'<mark style="background:{HIGHLIGHTS[category]};color:{INK};'
            f'padding:0 4px">{word}</mark>')


def masthead() -> str:
    today = date.today()
    return _flat(f"""
    <div class="mast">
      <div class="mast-bar">
        <span>Vol. I &middot; No. {today.timetuple().tm_yday}</span>
        <span>{today.strftime('%A, %B %d, %Y')}</span>
        <span><a href="{REPO_URL}" target="_blank">GitHub &#8599;</a></span>
      </div>
      <div class="mast-title">The Bias Gazette</div>
      <div class="mast-sub">News Bias &amp; Framing Analyzer &nbsp;&middot;&nbsp; Web &amp; Social Media Analytics</div>
    </div>
    """)


def hero() -> str:
    specimen = (
        f"The {_mark('reckless', 'Negative loaded')} government "
        f"{_mark('slammed', 'Negative loaded')} farmers with a "
        f"{_mark('shocking', 'Negative loaded')} new law, while supporters "
        f"{_mark('hailed', 'Positive loaded')} it as a {_mark('bold', 'Positive loaded')} "
        f"and {_mark('historic', 'Positive loaded')} reform."
    )
    return _flat(f"""
    <div class="hero">
      <div>
        <div class="kicker">A news framing laboratory</div>
        <div class="hero-h">Same story. <em>Different</em> spin.</div>
        <div class="hero-p">Paste an article, drop in a link, or compare three outlets covering the same event.
        The analyzer measures sentiment, vocabulary, loaded language and how each person or organisation
        is portrayed, then shows its working.</div>
        <div class="pills">
          <span class="pill">VADER + TextBlob</span>
          <span class="pill">Lexical diversity</span>
          <span class="pill">Entity-level framing</span>
          <span class="pill">Transparent bias score</span>
        </div>
      </div>
      <div class="clip">
        <div class="clip-lab">Specimen no. 1 &middot; fictional sentence</div>
        <div class="clip-txt">{specimen}</div>
        <div class="clip-foot">Every highlight above is flagged automatically by the loaded-word lexicon.</div>
      </div>
    </div>
    """)


def stat_cards(items: list) -> str:
    """items = [(label, big number, description), ...]"""
    cards = "".join(
        f'<div class="stat"><div class="lab">{escape(lab)}</div>'
        f'<div class="num">{escape(num)}</div><div class="txt">{escape(txt)}</div></div>'
        for lab, num, txt in items
    )
    return f'<div class="stats">{cards}</div>'


def card(label: str, title: str, body: str) -> str:
    return _flat(f'<div class="card"><div class="lab">{escape(label)}</div>'
                 f'<div class="ctitle">{escape(title)}</div><div class="txt">{escape(body)}</div></div>')


def section(kicker: str, title: str) -> str:
    return _flat(f'<div class="sec"><div class="kicker">{escape(kicker)}</div>'
                 f'<div class="sec-t">{escape(title)}</div></div>')


def callout(tag: str, body_html: str) -> str:
    return _flat(f'<div class="callout"><span class="tag">{escape(tag)}</span>{body_html}</div>')


def footer() -> str:
    return _flat("""
    <div class="foot">Built with Python &middot; NLTK &middot; spaCy &middot; VADER &middot; TextBlob &middot; Plotly &middot; Streamlit<br>
    A heuristic indicator of writing style. Not a fact-check and not a political label.</div>
    """)