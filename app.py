"""Fieldwise: crop recommendation web app (Streamlit).
Run:  streamlit run app.py
Needs the ./artifacts folder produced by training_notebook.ipynb.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artifacts"
LOW_PROB_THRESHOLD = 0.50   # set from notebook section 10, then report the value
FAR_MARGIN = 0.20           # "far outside training range" = beyond 20% of the range past min/max

# Hard limits (impossible values) and display info: (label, unit, hard_min, hard_max, step)
FIELDS = {
    "N": ("Nitrogen (N)", "", 0.0, 300.0, 1.0),
    "P": ("Phosphorus (P)", "", 0.0, 300.0, 1.0),
    "K": ("Potassium (K)", "", 0.0, 400.0, 1.0),
    "ph": ("Soil pH", "", 0.0, 14.0, 0.1),
    "temperature": ("Temperature", "°C", -10.0, 60.0, 0.5),
    "humidity": ("Humidity", "%", 0.0, 100.0, 1.0),
    "rainfall": ("Rainfall", "mm", 0.0, 500.0, 1.0),
}
SOIL = ["N", "P", "K", "ph"]
WEATHER = ["temperature", "humidity", "rainfall"]


# ---------- logic (no UI) ----------
def load_artifacts():
    model = joblib.load(ART / "model.joblib")
    scaler = joblib.load(ART / "scaler.joblib")
    encoder = joblib.load(ART / "label_encoder.joblib")
    ranges = json.loads((ART / "training_ranges.json").read_text(encoding="utf-8"))
    profiles = json.loads((ART / "crop_profiles.json").read_text(encoding="utf-8"))
    return model, scaler, encoder, ranges, profiles


def validate(values):
    """Return {field: message} for missing or impossible readings."""
    errors = {}
    for key, (label, unit, lo, hi, _) in FIELDS.items():
        v = values.get(key)
        if v is None:
            errors[key] = "Enter a value."
        elif not (lo <= v <= hi):
            errors[key] = f"{label} must be between {lo:g} and {hi:g}{(' ' + unit) if unit else ''}."
    return errors


def unusual_fields(values, ranges):
    """Fields far outside the range seen in training."""
    out = []
    for key, r in ranges["ranges"].items():
        span = max(r["max"] - r["min"], 1e-9)
        v = values[key]
        if v < r["min"] - FAR_MARGIN * span or v > r["max"] + FAR_MARGIN * span:
            out.append(key)
    return out


def predict_top3(values, model, scaler, encoder, ranges):
    cols = ranges["features"]
    X = pd.DataFrame([[values[c] for c in cols]], columns=cols)
    if "clip_lower" in ranges and "clip_upper" in ranges:
        X = X.clip(lower=pd.Series(ranges["clip_lower"]), upper=pd.Series(ranges["clip_upper"]), axis=1)
    Xs = pd.DataFrame(scaler.transform(X), columns=cols)
    proba = model.predict_proba(Xs)[0]
    names = encoder.inverse_transform(model.classes_)
    order = proba.argsort()[::-1][:3]
    return [(names[i], float(proba[i])) for i in order]


def fit_table(values, crop, profiles):
    rows = []
    for key in SOIL + WEATHER:
        p = profiles[crop][key]
        v = values[key]
        status = "within typical range" if p["low"] <= v <= p["high"] else ("above typical range" if v > p["high"] else "below typical range")
        rows.append({"Reading": FIELDS[key][0], "Your value": v,
                     "Typical range": f"{p['low']:.1f} to {p['high']:.1f}", "Fit": status})
    return pd.DataFrame(rows)


# ---------- UI ----------
st.set_page_config(page_title="Fieldwise", page_icon="🌱", layout="centered")
st.markdown("""
<style>
.stApp {background:#f7f6ee;}
h1,h2,h3 {font-family: Georgia, 'Times New Roman', serif; color:#143d2e;}
.top-card {background:#143d2e; color:#fff; border-radius:14px; padding:22px 26px; margin-bottom:12px;}
.top-card .tag {color:#cfee8a; font-size:0.75rem; letter-spacing:.08em;}
.top-card .crop {font-family: Georgia, serif; font-size:2.8rem; line-height:1.1; margin:6px 0;}
div.stButton > button[kind="primary"] {background:#235c43; border:none; color:#fff;}
</style>
""", unsafe_allow_html=True)

try:
    MODEL, SCALER, ENCODER, RANGES, PROFILES = st.cache_resource(load_artifacts)()
except FileNotFoundError:
    st.title("Assessment unavailable")
    st.error("Run the export section of training_notebook.ipynb to generate the model files in artifacts/.")
    if st.button("Retry loading"):
        st.rerun()
    st.stop()

ss = st.session_state
ss.setdefault("stage", "welcome")      # welcome | inputs | warn | results
ss.setdefault("values", {})
ss.setdefault("errors", {})
ss.setdefault("unusual", [])

def go(stage):
    ss.stage = stage


def sample_values(crop):
    return {k: round(PROFILES[crop][k]["median"], 1) for k in FIELDS}


@st.dialog("Field guide")
def field_guide():
    st.write("Use nutrient readings (N, P, K) and pH from your soil test. "
             "Use temperature, humidity and rainfall for the same field. "
             "Enter values in the same units shown next to each box.")
    if st.button("Got it"):
        st.rerun()


@st.dialog("Start a new field?")
def new_field_dialog():
    st.write("This clears the seven readings and returns you to a blank form.")
    c1, c2 = st.columns(2)
    if c1.button("Clear readings", type="primary"):
        for k in FIELDS:
            ss.pop(f"in_{k}", None)
        ss["values"], ss.errors, ss.unusual = {}, {}, []
        go("inputs"); st.rerun()
    if c2.button("Cancel"):
        st.rerun()


@st.dialog("Why this crop?")
def why_dialog(crop):
    st.write("The model compares your seven readings with what it learned for each crop. "
             "This table shows how your readings sit against the typical range for this crop.")
    st.dataframe(fit_table(ss["values"], crop, PROFILES), hide_index=True)
    if st.button("Got it"):
        st.rerun()


# Header
c1, c2 = st.columns([4, 1])
c1.markdown("### 🌱 Fieldwise")
if c2.button("Field guide"):
    field_guide()

# ---- Welcome ----
if ss.stage == "welcome":
    st.caption("CROP RECOMMENDATION")
    st.title("What grows well here?")
    st.write("Use your soil and climate readings to find crops suited to your field.")
    if st.button("Assess my field →", type="primary"):
        go("inputs"); st.rerun()
    st.divider()
    st.subheader("How it works")
    a, b, c = st.columns(3)
    a.write("1. Enter seven field readings")
    b.write("2. Compare 22 crop classes")
    c.write("3. Review your top three")

# ---- Inputs ----
elif ss.stage in ("inputs", "warn"):
    st.caption("FIELD ASSESSMENT")
    st.title("Field conditions")
    st.write("Use soil and weather readings from the same field.")

    with st.expander("Example field"):
        choice = st.selectbox("Preview with a sample", ["Typical field (first crop)", "High-pH field"] + [f"Typical {c}" for c in PROFILES])
        if st.button("Choose sample →", disabled=ss.stage == "warn"):
            if choice == "High-pH field":
                vals = sample_values(list(PROFILES)[0]); vals["ph"] = 14.0
            elif choice.startswith("Typical field"):
                vals = sample_values(list(PROFILES)[0])
            else:
                vals = sample_values(choice.replace("Typical ", "", 1))
            for k, v in vals.items():
                ss[f"in_{k}"] = float(v)
            ss.errors = {}; go("inputs"); st.rerun()

    def field_box(key):
        label, unit, lo, hi, step = FIELDS[key]
        r = RANGES["ranges"][key]
        shown = f"{label} ({unit})" if unit else label
        st.number_input(shown, value=None, step=step, key=f"in_{key}",
                        disabled=ss.stage == "warn",
                        placeholder=f"{r['min']:.0f} to {r['max']:.0f}",
                        help=f"Allowed {lo:g} to {hi:g}. Training data covered {r['min']:.1f} to {r['max']:.1f}.")
        if key in ss.errors:
            st.error(ss.errors[key])

    st.subheader("Soil measurements")
    cols = st.columns(4)
    for col, key in zip(cols, SOIL):
        with col:
            field_box(key)
    st.subheader("Weather measurements")
    cols = st.columns(3)
    for col, key in zip(cols, WEATHER):
        with col:
            field_box(key)

    if ss.stage == "warn":
        bad = ", ".join(FIELDS[k][0] for k in ss.unusual)
        st.warning(f"Unusual readings: {bad}. These are far outside the range the model learned from, "
                   "so the recommendation may be unreliable.")
        w1, w2 = st.columns(2)
        if w1.button("View results"):
            go("results"); st.rerun()
        if w2.button("Edit readings"):
            go("inputs"); st.rerun()
    else:
        if st.button("Recommend crop →", type="primary"):
            vals = {k: ss.get(f"in_{k}") for k in FIELDS}
            ss.errors = validate(vals)
            if ss.errors:
                st.rerun()
            ss["values"] = vals
            ss.unusual = unusual_fields(vals, RANGES)
            go("warn" if ss.unusual else "results"); st.rerun()

# ---- Results ----
elif ss.stage == "results":
    top3 = predict_top3(ss["values"], MODEL, SCALER, ENCODER, RANGES)
    best, p_best = top3[0]
    st.caption("ASSESSMENT RESULTS")
    st.title("Your crop matches")

    if ss.unusual:
        st.warning("Unusual field values may reduce reliability.")
    if p_best < LOW_PROB_THRESHOLD:
        st.warning("Conditions fit several crops; confirm with an agricultural expert.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f'<div class="top-card"><div class="tag">TOP MATCH</div>'
                    f'<div class="crop">{best.title()}</div>'
                    f'<div>{p_best:.0%} model probability</div></div>', unsafe_allow_html=True)
        if st.button("Why this crop?"):
            why_dialog(best)
    with right:
        st.subheader("Top three crops")
        for name, p in top3:
            st.write(f"**{name.title()}**  {p:.0%}")
            st.progress(min(max(p, 0.0), 1.0))

    st.caption("Percentages are model probabilities across the 22 crop classes. "
               "Recommendation is advisory.")
    b1, b2 = st.columns(2)
    if b1.button("Edit readings"):
        for key, value in ss["values"].items():
            ss[f"in_{key}"] = value
        go("inputs"); st.rerun()
    if b2.button("New field", type="primary"):
        new_field_dialog()
