# Smart Travel Destination Recommendation System
# Run locally:  streamlit run app.py
# Needs travel_dataset.csv in the same folder as this file.

import base64
import html
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).parent / "travel_dataset.csv"

# ---------------------------------------------------------------
# 1) Settings (all values the UI can send to the model)
# ---------------------------------------------------------------
BUDGET_OPTIONS = ["Low", "Medium", "High"]
DURATION_OPTIONS = {"3 days": 3, "1 week": 7, "2 weeks": 14, "1 month": 30}
WEATHER_OPTIONS = ["Hot", "Moderate", "Cold"]
ACTIVITY_OPTIONS = ["Beach", "Adventure", "History", "Nightlife",
                    "Nature", "Shopping", "Hiking"]

BUDGET_LEVEL = {"low": 1, "medium": 2, "high": 3}
# Max total trip cost (USD) accepted for each budget level
TRIP_CAPS = {"low": 800, "medium": 2500, "high": 6000}
# Typical daily cost per budget level (used only as a tie-breaker)
TYPICAL_COST = {"low": 70, "medium": 175, "high": 425}
# Dataset climate -> weather option shown in the UI
WEATHER_MAP = {"Tropical": "hot", "Desert": "hot", "Temperate": "moderate",
               "Mediterranean": "moderate", "Cold": "cold"}
# UI activity -> feature column (vibe column, or a word searched in the activities text)
ACT_COLS = {"beach": "a_beach", "adventure": "a_adventure", "history": "a_historic",
            "nightlife": "a_nightlife", "nature": "a_nature",
            "shopping": "a_shopping", "hiking": "a_hiking"}
# Weights from the project report
WEIGHTS = {"budget": 0.30, "activities": 0.30, "weather": 0.20, "duration": 0.20}

# One fixed, realistic climate per city (climate is a property of the city)
CITY_CLIMATE = {
    "Amsterdam, Netherlands": "Temperate", "Athens, Greece": "Mediterranean",
    "Auckland, New Zealand": "Temperate", "Bali, Indonesia": "Tropical",
    "Bangkok, Thailand": "Tropical", "Barcelona, Spain": "Mediterranean",
    "Budapest, Hungary": "Temperate", "Buenos Aires, Argentina": "Temperate",
    "Cairo, Egypt": "Desert", "Cape Town, South Africa": "Mediterranean",
    "Cartagena, Colombia": "Tropical", "Chefchaouen, Morocco": "Mediterranean",
    "Chicago, USA": "Cold", "Copenhagen, Denmark": "Temperate",
    "Cusco, Peru": "Temperate", "Dubrovnik, Croatia": "Mediterranean",
    "Edinburgh, Scotland": "Temperate", "Fiji, Fiji": "Tropical",
    "Gizeh, Egypt": "Desert", "Goa, India": "Tropical",
    "Hobart, Australia": "Temperate", "Hoi An, Vietnam": "Tropical",
    "Kathmandu, Nepal": "Temperate", "Kigali, Rwanda": "Temperate",
    "Kyoto, Japan": "Temperate", "Lima, Peru": "Temperate",
    "Lisbon, Portugal": "Mediterranean", "Los Angeles, USA": "Mediterranean",
    "Luang Prabang, Laos": "Tropical", "Malé, Maldives": "Tropical",
    "Marrakesh, Morocco": "Desert", "Medellín, Colombia": "Temperate",
    "Melbourne, Australia": "Temperate", "Mexico City, Mexico": "Temperate",
    "Miami, USA": "Tropical", "Montevideo, Uruguay": "Temperate",
    "Montreal, Canada": "Cold", "Nairobi, Kenya": "Temperate",
    "New York, USA": "Temperate", "Paris, France": "Temperate",
    "Perth, Australia": "Mediterranean", "Prague, Czechia": "Temperate",
    "Quebec City, Canada": "Cold", "Queenstown, New Zealand": "Temperate",
    "Quito, Ecuador": "Temperate", "Reykjavik, Iceland": "Cold",
    "Rio de Janeiro, Brazil": "Tropical", "Rome, Italy": "Mediterranean",
    "San Francisco, USA": "Mediterranean", "Santiago, Chile": "Mediterranean",
    "Seoul, South Korea": "Temperate", "Seychelles, Seychelles": "Tropical",
    "Siem Reap, Cambodia": "Tropical", "Singapore, Singapore": "Tropical",
    "Sydney, Australia": "Temperate", "Tahiti, French Polynesia": "Tropical",
    "Taipei, Taiwan": "Tropical", "Tulum, Mexico": "Tropical",
    "Vancouver, Canada": "Temperate", "Zanzibar City, Tanzania": "Tropical",
}


# ---------------------------------------------------------------
# 2) Data loading and preparation (same steps as the notebook)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].str.strip()
    data["budget_level"] = data["budget"].map({"$": 1, "$$": 2, "$$$": 3})
    data["climate"] = data["name"].map(CITY_CLIMATE).fillna(data["climate"])
    data["weather"] = data["climate"].map(WEATHER_MAP)

    # Activity features: the vibe column + words searched in the activities text
    acts = pd.get_dummies(data["vibe"].str.lower(), prefix="a").astype(float)
    text = data["activities"].str.lower()
    acts["a_shopping"] = text.str.contains("shopping").astype(float)
    acts["a_hiking"] = text.str.contains("hiking").astype(float)
    for col in ACT_COLS.values():          # make sure every column exists
        if col not in acts:
            acts[col] = 0.0
    return data, acts


df, acts = load_data()


# ---------------------------------------------------------------
# 3) Recommendation model (Weighted Scoring)
# ---------------------------------------------------------------
def recommend(budget, duration, activities, weather, top_n=5):
    """Return the top_n destinations as a DataFrame with a match percentage."""
    bl = budget.lower()
    days = DURATION_OPTIONS[duration]
    result = df.copy()

    # Budget: 1 if same level, 0.5 if one level away, otherwise 0
    diff = (result["budget_level"] - BUDGET_LEVEL[bl]).abs()
    result["budget_s"] = (1 - diff * 0.5).clip(lower=0)

    # Duration: total trip cost must stay under the cap of the chosen budget
    total_cost = result["typical_cost_per_day_usd"] * days
    result["duration_s"] = (TRIP_CAPS[bl] / total_cost).clip(upper=1)

    # Weather: exact match with the chosen weather
    result["weather_s"] = (result["weather"] == weather.lower()).astype(float)

    # Activities: 1 if any chosen activity matches this destination
    hit = pd.Series(False, index=result.index)
    for act in activities:
        hit |= acts[ACT_COLS[act.lower()]] == 1
    result["act_s"] = hit.astype(float)

    result["match_pct"] = (100 * (
        WEIGHTS["budget"] * result["budget_s"]
        + WEIGHTS["activities"] * result["act_s"]
        + WEIGHTS["weather"] * result["weather_s"]
        + WEIGHTS["duration"] * result["duration_s"])).round(1)

    # Tie-breaker: daily cost close to the typical cost of the chosen budget
    closeness = (1 - (result["typical_cost_per_day_usd"] - TYPICAL_COST[bl]).abs()
                 / TYPICAL_COST[bl]).clip(lower=0)
    result["rank_score"] = result["match_pct"] + 0.1 * closeness

    # The dataset has several rows per destination: keep the best one
    result = result.sort_values("rank_score", ascending=False).drop_duplicates("name")
    result["days"] = days
    result["trip_cost"] = result["typical_cost_per_day_usd"] * days
    result["matched"] = [
        [x for x in activities if acts.loc[i, ACT_COLS[x.lower()]] == 1]
        for i in result.index
    ]
    return result.head(top_n)


# ---------------------------------------------------------------
# 4) Look and feel (background photo + styling)
# ---------------------------------------------------------------
# Put a file called background.jpg next to app.py to use your own photo.
# If it is missing, the online photo below is used (and a colour gradient if that fails).
BG_FILE = Path(__file__).parent / "background.jpg"
BG_URL = ("https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
          "?auto=format&fit=crop&w=1920&q=80")


def background_source():
    if BG_FILE.exists():
        encoded = base64.b64encode(BG_FILE.read_bytes()).decode()
        return f"data:image/jpeg;base64,{encoded}"
    return BG_URL


CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #16384a;
    background-image: linear-gradient(rgba(8,24,38,.40), rgba(8,24,38,.70)), url("__BG__");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

.hero { text-align: center; padding: 2.2rem 0 1.4rem 0; }
.hero .title {
    color: #ffffff; font-size: 2.7rem; font-weight: 700; line-height: 1.15;
    text-shadow: 0 2px 14px rgba(0,0,0,.55);
}
.hero .subtitle {
    color: #f1f5f9; font-size: 1.1rem; margin-top: .6rem;
    text-shadow: 0 1px 8px rgba(0,0,0,.6);
}

[data-testid="stForm"] {
    background: rgba(255,255,255,.93);
    border: none; border-radius: 20px; padding: 1.6rem;
    box-shadow: 0 10px 30px rgba(0,0,0,.30);
}
[data-testid="stForm"] label p { color: #1f2937 !important; font-weight: 600; }

.section-title {
    color: #ffffff; font-size: 1.7rem; font-weight: 700; margin: 2rem 0 .2rem 0;
    text-shadow: 0 2px 10px rgba(0,0,0,.6);
}
.section-sub { color: #e2e8f0; margin-bottom: 1rem; text-shadow: 0 1px 6px rgba(0,0,0,.6); }

.card {
    background: rgba(255,255,255,.95); border-radius: 18px 18px 0 0;
    padding: 1.1rem 1.3rem .9rem 1.3rem; margin-top: 1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,.28);
}
.card-top { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; }
.card .rank { color: #e07a3f; font-weight: 700; font-size: .9rem; }
.card .name { color: #12263a; font-size: 1.35rem; font-weight: 700; }
.card .meta { color: #5b6b7b; font-size: .92rem; margin-top: .15rem; }
.card .score { text-align: right; white-space: nowrap; }
.card .pct { color: #12263a; font-size: 1.35rem; font-weight: 700; }
.card .stars { color: #f5b301; font-size: 1.05rem; letter-spacing: 2px; }
.card .bar { background: #e5e9ee; border-radius: 99px; height: 8px; margin: .8rem 0 .7rem 0; overflow: hidden; }
.card .bar div { background: linear-gradient(90deg, #e07a3f, #f5b301); height: 100%; border-radius: 99px; }
.card .cost { color: #12263a; font-size: .95rem; }

[data-testid="stExpander"] {
    background: rgba(255,255,255,.95); border: none;
    border-radius: 0 0 18px 18px; margin-top: -1px;
    box-shadow: 0 8px 24px rgba(0,0,0,.28);
}
</style>
"""

st.set_page_config(page_title="Smart Travel Recommender", page_icon="🌍")
st.markdown(CSS.replace("__BG__", background_source()), unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="title">🌍 Find your perfect destination</div>'
    '<div class="subtitle">Tell us about your trip and we will suggest the places '
    'that fit you best.</div></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# 5) Questions
# ---------------------------------------------------------------
# Daily cost range of each budget level, taken from the dataset itself
ranges = df.groupby("budget_level")["typical_cost_per_day_usd"].agg(["min", "max"])
budget_help = " | ".join(
    f"{name}: ${int(ranges.loc[lvl, 'min'])}-{int(ranges.loc[lvl, 'max'])} per day"
    for name, lvl in zip(["Low", "Medium", "High"], [1, 2, 3])
)

with st.form("preferences"):
    c1, c2 = st.columns(2)
    budget = c1.selectbox("Budget", BUDGET_OPTIONS, index=1, help=budget_help)
    duration = c2.selectbox("Trip duration", list(DURATION_OPTIONS), index=1)
    weather = c1.selectbox("Preferred weather", WEATHER_OPTIONS, index=1,
                           help="Hot: tropical and desert. Moderate: temperate and "
                                "Mediterranean. Cold: cold climates.")
    activities = c2.multiselect("What do you love to do? (one or more)",
                                ACTIVITY_OPTIONS, default=["History"])
    submitted = st.form_submit_button("Find my destinations", type="primary")

# ---------------------------------------------------------------
# 6) Results
# ---------------------------------------------------------------
def result_card(rank, r):
    """HTML for one result card."""
    esc = html.escape
    stars = round(r["match_pct"] / 20)
    meta = []
    if r["matched"]:
        meta.append("Matches: " + ", ".join(r["matched"]))
    meta.append(f"{r['climate']} climate")
    return (
        '<div class="card"><div class="card-top"><div>'
        f'<div class="rank">#{rank}</div>'
        f'<div class="name">{esc(r["name"])}</div>'
        f'<div class="meta">{esc(" | ".join(meta))}</div></div>'
        '<div class="score">'
        f'<div class="pct">{r["match_pct"]:.0f}% match</div>'
        f'<div class="stars">{"★" * stars}{"☆" * (5 - stars)}</div></div></div>'
        f'<div class="bar"><div style="width:{min(r["match_pct"], 100):.0f}%"></div></div>'
        f'<div class="cost"><b>Estimated trip cost:</b> about ${int(r["trip_cost"]):,} '
        f'for {int(r["days"])} days (${int(r["typical_cost_per_day_usd"])}/day)</div></div>'
    )


if submitted:
    if not activities:
        st.warning("Please choose at least one activity.")
    else:
        results = recommend(budget, duration, activities, weather)
        st.markdown('<div class="section-title">Top recommendations</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-sub">{budget} budget | {duration} | {weather} weather | '
            f'{html.escape(", ".join(activities))}</div>',
            unsafe_allow_html=True,
        )
        for rank, (_, r) in enumerate(results.iterrows(), 1):
            st.markdown(result_card(rank, r), unsafe_allow_html=True)
            with st.expander("Details"):
                st.write(r["description"])
                st.write(f"**Trip style:** {r['vibe']}  |  **Best for:** {r['group_size']}")
                st.write(f"**Best months:** {r['best_months']}")
                st.write(f"**Things to do:** {r['activities']}")

st.write("")
with st.expander("How the recommendation score works"):
    st.write(
        "Every destination gets a score out of 100: budget fit (30%), matching "
        "activities (30%), weather (20%) and trip cost for your duration (20%). "
        "Destinations with the same score are ordered by how close their daily "
        "cost is to the typical cost of your budget."
    )
