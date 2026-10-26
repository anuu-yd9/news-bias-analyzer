"""
samples.py
Three fictional articles about the SAME made-up event, written in different tones.
All people, organisations and quotes are invented for demonstration purposes.
Lets the dashboard demo work completely offline.
"""

STORY = "Metro fare hike in Bengaluru (fictional)"

SAMPLES = {
    "Daily Ledger (critical tone)": """
Bengaluru commuters were hit with a shocking fare hike on Monday after the Bengaluru
Urban Transit Authority rammed through a brutal 40 percent increase in metro tickets.
Chairman Rohan Mehta defended the reckless decision, claiming it was clearly unavoidable.
Critics say the move is a disaster for working families who already struggle with rising
rents. Meera Iyer of the Commuters' Forum blasted the authority for ignoring public outrage
and called the hike a betrayal of ordinary riders. Mehta insisted that the authority had no
choice, but experts say the authority has wasted money on inflated contracts for years.
The backlash is growing across Bengaluru, and many fear the decision will devastate daily
commuters. Opposition leaders slammed the authority's arrogant handling of the crisis and
demanded an immediate rollback. Sources say the board approved the hike behind closed doors,
with little regard for those who rely on the metro every day.
""",

    "City Wire (neutral tone)": """
The Bengaluru Urban Transit Authority approved a 40 percent increase in metro fares on
Monday. The new fares take effect on 1 December. Chairman Rohan Mehta said the authority had
not revised ticket prices in six years and that operating costs have risen over that period.
According to the authority, the additional revenue will fund maintenance and the purchase of
new trains. Meera Iyer of the Commuters' Forum said the group would submit a petition asking
for a review of the decision. Transport economist Sanjay Rao said the effect on ridership may
become clearer after three months of data. The board voted 7 to 2 in favour of the proposal.
A public hearing on the fare structure is scheduled for next month. Monthly pass holders will
receive a discount of 15 percent under the revised structure.
""",

    "Metro Today (supportive tone)": """
In a bold and historic move, the Bengaluru Urban Transit Authority approved a landmark fare
reform on Monday that promises to boost the city's metro network for decades. Chairman Rohan
Mehta praised the decision as visionary, saying the extra revenue will deliver impressive
upgrades, including brand new trains and safer stations. Transport economist Sanjay Rao hailed
the plan as a remarkable breakthrough that will help the system thrive. Commuters will soon
enjoy more frequent services and a stunning improvement in comfort. Meera Iyer of the
Commuters' Forum raised concerns, but Mehta pledged to work with riders and celebrated the
board's courageous vote. Experts say the reform is clearly the right choice, and supporters
across Bengaluru have welcomed the decision. The discount for monthly pass holders is a
thoughtful touch that shows real commitment to ordinary riders.
""",
}


if __name__ == "__main__":
    import pandas as pd
    from bias import analyze_article

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    for name, text in SAMPLES.items():
        res = analyze_article(text, name)
        print("=" * 80)
        print(f"{name}")
        print(f"  Bias score : {res['bias']['score']} ({res['bias']['label']})")
        print(f"  Tone       : {res['bias']['tone']}")
        print(f"  VADER      : {res['overall']['vader_compound']:.2f}")
        print(f"  Subjectivity: {res['overall']['textblob_subjectivity']:.2f}")
        print(f"  MTLD       : {res['lexical']['mtld']:.1f}")
        print(res["entity_framing"][["Entity", "Type", "Mentions", "Avg Sentiment", "Portrayal"]]
              .to_string(index=False))