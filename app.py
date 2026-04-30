import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Travel Insurance Claim Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS — PREMIUM BUSINESS REDESIGN ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
}
.stApp {
    background: #080F1E;
    color: #E2E8F8;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0C1526 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stMarkdown { color: #C0CCDF; }

[data-testid="stSidebar"] label {
    color: #7A8FAD !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 4px !important;
}

/* Sidebar selectbox & number inputs */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    color: #D4DCEF !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:hover,
[data-testid="stSidebar"] .stNumberInput > div > div > input:hover {
    border-color: rgba(99,148,255,0.35) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
[data-testid="stSidebar"] .stNumberInput > div > div > input:focus {
    border-color: rgba(99,148,255,0.65) !important;
    box-shadow: 0 0 0 3px rgba(99,148,255,0.12) !important;
}

/* Sidebar number input +/- buttons */
[data-testid="stSidebar"] .stNumberInput button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 6px !important;
    color: #7A8FAD !important;
}
[data-testid="stSidebar"] .stNumberInput button:hover {
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(99,148,255,0.3) !important;
}

/* ── Main content ── */
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── HEADER BANNER — Full-width, impactful ── */
.header-banner {
    background: linear-gradient(135deg, #06111F 0%, #0E2040 45%, #0A1930 100%);
    border-bottom: 1px solid rgba(99,148,255,0.18);
    padding: 36px 44px 30px 44px;
    margin: -1rem -4rem 32px -4rem;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(99,148,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.header-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 40%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(52,211,153,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.header-eyebrow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
}
.h-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
}
.h-tag-blue  { background: rgba(99,148,255,0.12); color: #6394FF; border: 1px solid rgba(99,148,255,0.25); }
.h-tag-teal  { background: rgba(52,211,153,0.10); color: #34D399; border: 1px solid rgba(52,211,153,0.22); }
.h-tag-amber { background: rgba(251,191,36,0.10); color: #FBBF24; border: 1px solid rgba(251,191,36,0.22); }

.header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 36px;
    font-weight: 400;
    color: #FFFFFF;
    letter-spacing: -0.4px;
    line-height: 1.1;
    margin-bottom: 10px;
}
.header-subtitle {
    font-size: 14px;
    color: #7A8FAD;
    line-height: 1.6;
    max-width: 560px;
}

/* ── Section cards ── */
.section-card {
    background: #0D1829;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 24px 26px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .14em;
    color: #3D5070;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title-icon {
    width: 20px; height: 20px;
    border-radius: 5px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    flex-shrink: 0;
}
.icon-blue  { background: rgba(99,148,255,0.15); }
.icon-teal  { background: rgba(52,211,153,0.12); }
.icon-amber { background: rgba(251,191,36,0.12); }
.icon-rose  { background: rgba(251,113,133,0.12); }

/* ── Input summary grid ── */
.input-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}
.input-item {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 9px;
    padding: 12px 16px;
}
.input-label {
    color: #3A4F6A;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: 4px;
}
.input-value {
    color: #C4CDE0;
    font-size: 14px;
    font-weight: 500;
}

/* ── Model metric chips — bigger, more breathing room ── */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 18px;
}
.metric-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    flex: 1;
    min-width: 90px;
    transition: border-color .15s;
}
.metric-chip:hover {
    border-color: rgba(99,148,255,0.25);
}
.chip-label {
    color: #3D5070;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: 8px;
}
.chip-value {
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.3px;
}

/* ── Imbalance note ── */
.imbalance-note {
    background: rgba(251,191,36,0.05);
    border: 1px solid rgba(251,191,36,0.16);
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 13px;
    color: #9A8A50;
    line-height: 1.65;
}
.imbalance-note strong { color: #FBBF24; font-weight: 600; }

/* ── Feature engineering cards ── */
.fe-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}
.fe-item {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 14px 16px;
}
.fe-label {
    color: #3A4F6A;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .09em;
    margin-bottom: 6px;
}
.fe-value {
    color: #C4CDE0;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 5px;
}
.fe-badge-yes {
    display: inline-block;
    color: #FB7185;
    background: rgba(251,113,133,0.10);
    border: 1px solid rgba(251,113,133,0.22);
    border-radius: 5px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
}
.fe-badge-no {
    display: inline-block;
    color: #34D399;
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.20);
    border-radius: 5px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
}

/* ── RESULT CARD — High contrast, unmissable ── */
.result-claim {
    background: linear-gradient(145deg, #1C1200 0%, #241700 50%, #1A1000 100%);
    border: 2px solid rgba(251,191,36,0.45);
    border-radius: 16px;
    padding: 32px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-claim::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(251,191,36,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.result-no-claim {
    background: linear-gradient(145deg, #001810 0%, #001E14 50%, #001408 100%);
    border: 2px solid rgba(52,211,153,0.40);
    border-radius: 16px;
    padding: 32px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-no-claim::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(52,211,153,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.result-icon-wrap {
    width: 60px; height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px auto;
    font-size: 26px;
}
.icon-wrap-claim    { background: rgba(251,191,36,0.12); border: 2px solid rgba(251,191,36,0.28); }
.icon-wrap-no-claim { background: rgba(52,211,153,0.10); border: 2px solid rgba(52,211,153,0.24); }

.result-eyebrow {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .15em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.result-eyebrow-claim    { color: #7A6320; }
.result-eyebrow-no-claim { color: #1E6644; }

.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 32px;
    font-weight: 400;
    margin-bottom: 10px;
    letter-spacing: -0.3px;
}
.result-label-claim    { color: #FBBF24; }
.result-label-no-claim { color: #34D399; }

.risk-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 6px 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.risk-badge-veryhigh { background: rgba(251,113,133,0.15); color: #FB7185; border: 1px solid rgba(251,113,133,0.30); }
.risk-badge-high     { background: rgba(251,191,36,0.15);  color: #FBBF24; border: 1px solid rgba(251,191,36,0.30); }
.risk-badge-medium   { background: rgba(251,191,36,0.12);  color: #F59E0B; border: 1px solid rgba(251,191,36,0.25); }
.risk-badge-low      { background: rgba(52,211,153,0.10);  color: #34D399; border: 1px solid rgba(52,211,153,0.25); }

/* ── GAUGE — Larger, bolder ── */
.gauge-container {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 22px 18px 16px;
    margin-top: 18px;
}
.gauge-header {
    text-align: center;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .13em;
    color: #3D5070;
    margin-bottom: 10px;
}
.gauge-legend {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.gauge-leg-item {
    font-size: 11px;
    display: flex;
    align-items: center;
    gap: 5px;
    color: #6B7A99;
}
.leg-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

/* ── Threshold info ── */
.threshold-info {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 18px;
    margin-top: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: #7A8FAD;
}
.threshold-val {
    color: #6394FF;
    font-weight: 700;
    font-size: 15px;
}

/* ── Risk factors ── */
.risk-factor-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 7px;
    padding: 10px 14px;
    border-radius: 9px;
    font-size: 13px;
}
.rf-high { background: rgba(251,113,133,0.06); border: 1px solid rgba(251,113,133,0.16); }
.rf-low  { background: rgba(52,211,153,0.05);  border: 1px solid rgba(52,211,153,0.14); }
.rf-dot  { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }
.rf-dot-high { background: #FB7185; }
.rf-dot-low  { background: #34D399; }
.rf-name { font-weight: 600; color: #C4CDE0; margin-bottom: 2px; }
.rf-desc { color: #6B7A99; line-height: 1.4; font-size: 12px; }

/* ── Risk section label ── */
.risk-section-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .11em;
    margin-bottom: 10px;
    margin-top: 16px;
}
.rsl-high { color: #FB7185; }
.rsl-low  { color: #34D399; }

/* ── Error box ── */
.error-box {
    background: rgba(251,113,133,0.06);
    border: 1px solid rgba(251,113,133,0.25);
    border-radius: 12px;
    padding: 22px 26px;
    color: #FB7185;
    font-size: 14px;
    margin: 20px 0;
    line-height: 1.6;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1A3A8F, #2563EB) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: .04em !important;
    cursor: pointer !important;
    transition: opacity 0.18s, transform 0.1s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover { opacity: 0.87 !important; }
.stButton > button:active { transform: scale(0.98) !important; }

/* Reset button */
.stButton:last-of-type > button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #7A8FAD !important;
}
.stButton:last-of-type > button:hover {
    background: rgba(255,255,255,0.09) !important;
    color: #B0BDD4 !important;
}

/* ── Sidebar section header ── */
.sidebar-section {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .15em;
    color: #2E4060;
    padding: 8px 0 5px 0;
    margin-top: 6px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    gap: 7px;
}
.ss-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

/* ── Sidebar logo header ── */
.sidebar-logo {
    text-align: center;
    padding: 20px 0 14px 0;
    margin-bottom: 4px;
}
.sidebar-logo-icon {
    width: 48px; height: 48px;
    border-radius: 14px;
    background: rgba(99,148,255,0.10);
    border: 1px solid rgba(99,148,255,0.22);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 12px auto;
    font-size: 22px;
}
.sidebar-title {
    color: #D4DCEF;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.1px;
    margin-bottom: 3px;
}
.sidebar-subtitle {
    color: #3D5070;
    font-size: 11px;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 16px 0;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 50px 20px;
    color: #3D5070;
}
.empty-state-icon { font-size: 40px; margin-bottom: 14px; opacity: .6; }
.empty-state-text { font-size: 14px; line-height: 1.6; }
.empty-state-text strong { color: #6394FF; font-weight: 600; }

/* ── Result desc box ── */
.result-desc {
    margin-top: 16px;
    padding: 12px 16px;
    border-radius: 9px;
    font-size: 12px;
    line-height: 1.6;
    text-align: left;
}
.result-desc-claim    { background: rgba(0,0,0,0.3); border-left: 3px solid rgba(251,191,36,0.5); color: #8A7530; }
.result-desc-no-claim { background: rgba(0,0,0,0.3); border-left: 3px solid rgba(52,211,153,0.5); color: #2A7A50; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 28px 0 12px 0;
    color: #252E40;
    font-size: 12px;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 10px;
}
.footer span { color: #3D5070; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
MODEL_PATH = "model_travel_insurance_balanced.sav"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model_pkg = load_model(MODEL_PATH)

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div class="header-eyebrow">
        <span class="h-tag h-tag-blue">🤖 Logistic Regression Balanced</span>
        <span class="h-tag h-tag-teal">📊 Binary Classification</span>
        <span class="h-tag h-tag-amber">⚠️ Imbalanced Data Handling</span>
    </div>
    <div class="header-title">✈️ Travel Insurance Claim Predictor</div>
    <div class="header-subtitle">
        Prediksi potensi klaim pemegang polis berbasis Machine Learning —
        untuk pengambilan keputusan bisnis yang lebih tepat dan terukur.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── MODEL ERROR CHECK ─────────────────────────────────────────────────────────
if model_pkg is None:
    st.markdown(f"""
    <div class="error-box">
        <strong>⚠️ File Model Tidak Ditemukan</strong><br><br>
        File <code>{MODEL_PATH}</code> tidak ditemukan di direktori yang sama dengan <code>app.py</code>.<br><br>
        Pastikan struktur direktori Anda seperti berikut:<br>
        <pre style="margin-top:10px; color:#E2E8F8; background:rgba(255,255,255,0.04);
                    padding:14px; border-radius:8px; border: 1px solid rgba(255,255,255,0.07);">
├── app.py
├── requirements.txt
└── {MODEL_PATH}
        </pre>
        Jalankan notebook Capstone 3 terlebih dahulu untuk menghasilkan file model,
        kemudian letakkan file <code>.sav</code> tersebut di folder yang sama dengan <code>app.py</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "agency"               : "EPX",
    "agency_type"          : "Travel Agency",
    "distribution_channel" : "Online",
    "product_name"         : "Cancellation Plan",
    "gender"               : "Unknown",
    "age"                  : 35,
    "destination"          : "SINGAPORE",
    "duration"             : 10,
    "net_sales"            : 50.0,
    "commission"           : 12.5,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── EXTRACT MODEL COMPONENTS ──────────────────────────────────────────────────
preprocessor       = model_pkg["preprocessor"]
model              = model_pkg["model"]
threshold          = model_pkg["threshold"]
high_risk_dest_set = model_pkg["high_risk_destinations"]
num_features       = model_pkg["num_features"]
cat_features       = model_pkg["cat_features"]
metrics            = model_pkg["metrics"]
model_name         = model_pkg["model_name"]

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🛡️</div>
        <div class="sidebar-title">Input Pemegang Polis</div>
        <div class="sidebar-subtitle">Isi semua field di bawah ini</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # ── Agency Info ──
    st.markdown(
        '<div class="sidebar-section">'
        '<span class="ss-dot" style="background:#6394FF;"></span>'
        'Informasi Agen'
        '</div>',
        unsafe_allow_html=True
    )
    agency_options = [
        "EPX", "CWT", "JWT", "RAB", "CBH", "SSI", "KML", "C2B",
        "ADM", "LWC", "TST", "ART", "TTW", "JZI", "CCR", "CSR"
    ]
    agency = st.selectbox("Agency", agency_options,
        index=agency_options.index(st.session_state["agency"]), key="agency")

    agency_type = st.selectbox(
        "Agency Type", ["Airlines", "Travel Agency"],
        index=["Airlines", "Travel Agency"].index(st.session_state["agency_type"]),
        key="agency_type"
    )
    distribution_channel = st.selectbox(
        "Distribution Channel", ["Online", "Offline"],
        index=["Online", "Offline"].index(st.session_state["distribution_channel"]),
        key="distribution_channel"
    )

    # ── Product ──
    st.markdown(
        '<div class="sidebar-section" style="margin-top:20px;">'
        '<span class="ss-dot" style="background:#34D399;"></span>'
        'Produk Asuransi'
        '</div>',
        unsafe_allow_html=True
    )
    product_options = [
        "Cancellation Plan", "Comprehensive Plan", "Bronze Plan",
        "Silver Plan", "Gold Plan", "2 way Comprehensive Plan",
        "Value Plan", "Basic Plan", "Premier Plan",
        "Rental Vehicle Excess Insurance", "Annual Silver Plan",
        "Annual Gold Plan", "Annual Platinum Plan",
        "Single Trip Travel Protect Silver",
        "Single Trip Travel Protect Gold",
        "Single Trip Travel Protect Platinum",
        "Spouse or Parents Plan", "Child Comprehensive Plan",
        "Individual Comprehensive Plan", "Group Comprehensive Plan",
        "Travel Cruise Protect", "Travel Cruise Protect Family",
        "1 way Comprehensive Plan", "3 way Comprehensive Plan",
        "Ticket Protector", "Annual Travel Protect Silver"
    ]
    product_name = st.selectbox("Product Name", product_options,
        index=product_options.index(st.session_state["product_name"]), key="product_name")

    # ── Personal Info ──
    st.markdown(
        '<div class="sidebar-section" style="margin-top:20px;">'
        '<span class="ss-dot" style="background:#FB7185;"></span>'
        'Informasi Pemegang Polis'
        '</div>',
        unsafe_allow_html=True
    )
    gender = st.selectbox(
        "Gender", ["F", "M", "Unknown"],
        index=["F", "M", "Unknown"].index(st.session_state["gender"]),
        key="gender",
        help="Pilih 'Unknown' jika data gender tidak tersedia"
    )
    age = st.number_input(
        "Age (tahun)", min_value=0, max_value=120,
        value=st.session_state["age"], step=1, key="age",
        help="Usia pemegang polis (akan di-clip ke rentang 18–100)"
    )

    # ── Trip Info ──
    st.markdown(
        '<div class="sidebar-section" style="margin-top:20px;">'
        '<span class="ss-dot" style="background:#6394FF;"></span>'
        'Informasi Perjalanan'
        '</div>',
        unsafe_allow_html=True
    )
    destination_options = [
        "SINGAPORE", "MALAYSIA", "THAILAND", "INDONESIA", "AUSTRALIA",
        "JAPAN", "CHINA", "INDIA", "UNITED KINGDOM", "UNITED STATES",
        "HONG KONG", "TAIWAN", "SOUTH KOREA", "PHILIPPINES", "VIETNAM",
        "FRANCE", "GERMANY", "ITALY", "SPAIN", "NETHERLANDS",
        "CANADA", "NEW ZEALAND", "UNITED ARAB EMIRATES", "EGYPT",
        "SOUTH AFRICA", "BRAZIL", "ARGENTINA", "MEXICO", "TURKEY",
        "GREECE", "PORTUGAL", "SWITZERLAND", "AUSTRIA", "SWEDEN", "OTHER"
    ]
    destination = st.selectbox("Destination", destination_options,
        index=destination_options.index(st.session_state["destination"]), key="destination")
    duration = st.number_input(
        "Duration (hari)", min_value=0, max_value=730,
        value=st.session_state["duration"], step=1, key="duration",
        help="Durasi perjalanan dalam hari"
    )

    # ── Financial Info ──
    st.markdown(
        '<div class="sidebar-section" style="margin-top:20px;">'
        '<span class="ss-dot" style="background:#FBBF24;"></span>'
        'Informasi Finansial'
        '</div>',
        unsafe_allow_html=True
    )
    net_sales = st.number_input(
        "Net Sales (nilai premi)",
        value=st.session_state["net_sales"], step=1.0, format="%.2f", key="net_sales",
        help="Total nilai penjualan polis / premi"
    )
    commission = st.number_input(
        "Commission in Value", min_value=0.0,
        value=st.session_state["commission"], step=0.5, format="%.2f", key="commission",
        help="Komisi agen dalam nilai absolut"
    )

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Prediksi Klaim", use_container_width=True, type="primary")

    def reset_inputs():
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
    st.button("↩  Reset Input", use_container_width=True, on_click=reset_inputs)

# ─── MAIN LAYOUT ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:

    # ── Input Summary ──
    st.markdown("""
    <div class="section-card">
        <div class="section-title">
            <span class="section-title-icon icon-blue">📋</span>
            Ringkasan Input
        </div>
    """, unsafe_allow_html=True)

    input_data = {
        "Agency"               : agency,
        "Agency Type"          : agency_type,
        "Distribution Channel" : distribution_channel,
        "Product Name"         : product_name,
        "Gender"               : gender,
        "Age"                  : age,
        "Destination"          : destination.title(),
        "Duration"             : f"{duration} hari",
        "Net Sales"            : f"{net_sales:,.2f}",
        "Commission"           : f"{commission:,.2f}",
    }
    grid_html = '<div class="input-grid">'
    for label, value in input_data.items():
        grid_html += f"""
        <div class="input-item">
            <div class="input-label">{label}</div>
            <div class="input-value">{value}</div>
        </div>"""
    grid_html += '</div></div>'
    st.markdown(grid_html, unsafe_allow_html=True)

    # ── Model Info ──
    st.markdown("""
    <div class="section-card">
        <div class="section-title">
            <span class="section-title-icon icon-teal">🤖</span>
            Performa Model
        </div>
    """, unsafe_allow_html=True)

    recall_color    = "#34D399" if metrics['Recall']  >= 0.7  else ("#FBBF24" if metrics['Recall']  >= 0.5  else "#FB7185")
    precision_color = "#6B7A99"
    f1_color        = "#6394FF" if metrics['F1']      >= 0.15 else "#FBBF24"
    prauc_color     = "#6394FF" if metrics['PR AUC']  >= 0.10 else "#FBBF24"
    rocauc_color    = "#6394FF" if metrics['ROC AUC'] >= 0.70 else "#FBBF24"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-chip" title="Recall — metrik utama. Seberapa banyak klaim aktual yang berhasil terdeteksi.">
            <div class="chip-label">⭐ Recall</div>
            <div class="chip-value" style="color:{recall_color};">{metrics['Recall']:.2%}</div>
        </div>
        <div class="metric-chip" title="Precision rendah adalah wajar pada data imbalanced. Model diprioritaskan untuk Recall tinggi.">
            <div class="chip-label">Precision ⚠</div>
            <div class="chip-value" style="color:{precision_color};">{metrics['Precision']:.2%}</div>
        </div>
        <div class="metric-chip" title="F1-Score — harmonic mean antara Precision dan Recall.">
            <div class="chip-label">F1-Score</div>
            <div class="chip-value" style="color:{f1_color};">{metrics['F1']:.2%}</div>
        </div>
        <div class="metric-chip" title="PR AUC — lebih relevan dari ROC AUC untuk data sangat imbalanced.">
            <div class="chip-label">PR AUC</div>
            <div class="chip-value" style="color:{prauc_color};">{metrics['PR AUC']:.2%}</div>
        </div>
        <div class="metric-chip" title="ROC AUC — kemampuan model membedakan kelas Claim vs No Claim.">
            <div class="chip-label">ROC AUC</div>
            <div class="chip-value" style="color:{rocauc_color};">{metrics['ROC AUC']:.2%}</div>
        </div>
    </div>
    <div class="imbalance-note">
        <strong>⚠ Catatan Imbalanced Data</strong><br>
        Dataset klaim sangat tidak seimbang (<strong>~1.7% klaim</strong>).
        Model diprioritaskan untuk <strong>Recall tinggi</strong> agar klaim aktual tidak terlewat —
        konsekuensinya Precision menjadi rendah (trade-off yang disengaja).
    </div>
    </div>
    """, unsafe_allow_html=True)


with col_right:

    # ── Prediction Result ──
    st.markdown("""
    <div class="section-card">
        <div class="section-title">
            <span class="section-title-icon icon-amber">🎯</span>
            Hasil Prediksi
        </div>
    """, unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-text">
                Isi input di sidebar lalu klik<br>
                <strong>Prediksi Klaim</strong> untuk melihat hasil.
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Feature Engineering ──
        age_clipped      = np.clip(age, 18, 100)
        duration_clipped = max(duration, 0)

        age_group = pd.cut(
            [age_clipped],
            bins=[0, 25, 35, 45, 60, 101],
            labels=['<25', '25-35', '35-45', '45-60', '60+']
        ).astype(str)[0]

        duration_group = pd.cut(
            [duration_clipped],
            bins=[-1, 3, 7, 14, 30, 10000],
            labels=['1-3 hari', '4-7 hari', '8-14 hari', '15-30 hari', '>30 hari']
        ).astype(str)[0]

        commission_rate = (
            commission / abs(net_sales) if abs(net_sales) > 0 else 0
        )
        commission_rate = float(np.clip(commission_rate, 0, 1))

        is_long_trip          = 1 if duration_clipped > 14 else 0
        is_online             = 1 if distribution_channel == "Online" else 0
        high_risk_destination = 1 if destination in high_risk_dest_set else 0

        X_input = pd.DataFrame([{
            "Agency"               : agency,
            "Agency Type"          : agency_type,
            "Distribution Channel" : distribution_channel,
            "Product Name"         : product_name,
            "Gender"               : gender,
            "Duration"             : duration_clipped,
            "Destination"          : destination,
            "Net Sales"            : net_sales,
            "Commision (in value)" : commission,
            "Age"                  : age_clipped,
            "Age_Group"            : age_group,
            "Duration_Group"       : duration_group,
            "Commission_Rate"      : commission_rate,
            "Is_Long_Trip"         : is_long_trip,
            "Is_Online"            : is_online,
            "High_Risk_Destination": high_risk_destination,
        }])

        try:
            X_transformed = preprocessor.transform(X_input)
            prob          = float(model.predict_proba(X_transformed)[:, 1][0])
            pred_label    = "Claim" if prob >= threshold else "No Claim"
            is_claim      = pred_label == "Claim"
            prob_pct      = prob * 100

            # ── Risk level classification ──
            if prob_pct >= 70:
                risk_level = "VERY HIGH RISK"
                risk_desc  = "Risiko sangat tinggi — perlu evaluasi mendalam oleh tim underwriting."
                badge_cls  = "risk-badge-veryhigh"
                gc         = "#FB7185"
            elif prob_pct >= 40:
                risk_level = "MEDIUM RISK"
                risk_desc  = "Risiko sedang — disarankan pemantauan lebih lanjut."
                badge_cls  = "risk-badge-medium"
                gc         = "#FBBF24"
            elif prob_pct >= threshold * 100:
                risk_level = "LOW-MEDIUM RISK"
                risk_desc  = "Risiko di batas threshold — perlu perhatian standar."
                badge_cls  = "risk-badge-high"
                gc         = "#F59E0B"
            else:
                risk_level = "LOW RISK"
                risk_desc  = "Risiko rendah — pemegang polis kemungkinan tidak mengajukan klaim."
                badge_cls  = "risk-badge-low"
                gc         = "#34D399"

            # ── Result card ──
            pred_display  = "CLAIM" if is_claim else "NO CLAIM"
            card_cls      = "result-claim" if is_claim else "result-no-claim"
            icon_cls      = "icon-wrap-claim" if is_claim else "icon-wrap-no-claim"
            label_cls     = "result-label-claim" if is_claim else "result-label-no-claim"
            eye_cls       = "result-eyebrow-claim" if is_claim else "result-eyebrow-no-claim"
            desc_cls      = "result-desc-claim" if is_claim else "result-desc-no-claim"
            icon_emoji    = "⚠️" if is_claim else "✅"
            pred_sublabel = "Pemegang polis berpotensi mengajukan klaim." if is_claim else "Pemegang polis kemungkinan tidak mengajukan klaim."

            st.markdown(f"""
            <div class="{card_cls}">
                <div class="result-icon-wrap {icon_cls}">{icon_emoji}</div>
                <div class="result-eyebrow {eye_cls}">Prediksi Status Polis</div>
                <div class="result-label {label_cls}">{pred_display}</div>
                <div><span class="risk-badge {badge_cls}">{risk_level}</span></div>
                <div class="result-desc {desc_cls}">{risk_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── GAUGE CHART — Larger & bolder ──
            import math

            cx, cy, r_outer, r_inner = 150, 130, 100, 72

            def polar(angle_deg, radius=r_outer):
                rad = math.radians(angle_deg)
                return cx + radius * math.cos(rad), cy - radius * math.sin(rad)

            def arc_path(start_deg, end_deg, radius=r_outer, inner=r_inner):
                sx, sy   = polar(start_deg, radius)
                ex, ey   = polar(end_deg,   radius)
                six, siy = polar(start_deg, inner)
                eix, eiy = polar(end_deg,   inner)
                large = 1 if abs(start_deg - end_deg) > 180 else 0
                return (
                    f"M {sx:.2f} {sy:.2f} "
                    f"A {radius} {radius} 0 {large} 0 {ex:.2f} {ey:.2f} "
                    f"L {eix:.2f} {eiy:.2f} "
                    f"A {inner} {inner} 0 {large} 1 {six:.2f} {siy:.2f} Z"
                )

            def p2a(p): return 180 - (p / 100) * 180

            # Track background
            track_path = arc_path(180, 0)

            # Zone fills (subtle background)
            zones = [(0, 40, "#34D399"), (40, 70, "#FBBF24"), (70, 100, "#FB7185")]
            zone_paths = ""
            for zs, ze, zc in zones:
                zone_paths += f'<path d="{arc_path(p2a(zs), p2a(ze))}" fill="{zc}" opacity="0.12"/>'

            # Active arc fill
            needle_angle = p2a(prob_pct)
            if prob_pct > 0:
                active_arc = f'<path d="{arc_path(180, needle_angle)}" fill="{gc}" opacity="0.75"/>'
            else:
                active_arc = ""

            # Needle
            nad = math.radians(needle_angle)
            nx  = cx + 88 * math.cos(nad)
            ny  = cy - 88 * math.sin(nad)

            # Tick marks
            ticks_svg = ""
            for tv, tl in [(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")]:
                ta = p2a(tv)
                ox, oy = polar(ta, r_outer + 5)
                ix, iy = polar(ta, r_outer - 5)
                lx, ly = polar(ta, r_outer + 18)
                ticks_svg += (
                    f'<line x1="{ix:.1f}" y1="{iy:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" '
                    f'stroke="#2E4060" stroke-width="2"/>'
                )
                ticks_svg += (
                    f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                    f'dominant-baseline="middle" fill="#3D5070" '
                    f'font-size="9" font-family="DM Sans,sans-serif">{tl}</text>'
                )

            # Threshold marker
            ta2 = p2a(threshold * 100)
            ox2, oy2 = polar(ta2, r_outer + 8)
            ix2, iy2 = polar(ta2, r_inner - 8)
            lx2, ly2 = polar(ta2, r_outer + 24)
            ticks_svg += (
                f'<line x1="{ix2:.1f}" y1="{iy2:.1f}" x2="{ox2:.1f}" y2="{oy2:.1f}" '
                f'stroke="#6394FF" stroke-width="2" stroke-dasharray="4,3"/>'
            )
            ticks_svg += (
                f'<text x="{lx2:.1f}" y="{ly2:.1f}" text-anchor="middle" '
                f'fill="#6394FF" font-size="8" font-family="DM Sans,sans-serif" '
                f'font-weight="700">THR</text>'
            )

            gauge_html = f"""
            <div class="gauge-container">
                <div class="gauge-header">Probabilitas Klaim</div>
                <svg viewBox="0 0 300 170" xmlns="http://www.w3.org/2000/svg"
                     style="width:100%; max-width:340px; display:block; margin:0 auto;">
                    <path d="{track_path}" fill="#0A1528" stroke="rgba(255,255,255,0.05)" stroke-width="0.5"/>
                    {zone_paths}
                    {active_arc}
                    {ticks_svg}
                    <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}"
                          stroke="{gc}" stroke-width="4" stroke-linecap="round"/>
                    <circle cx="{cx}" cy="{cy}" r="10" fill="{gc}" opacity="0.95"/>
                    <circle cx="{cx}" cy="{cy}" r="5"  fill="#080F1E"/>
                    <text x="{cx}" y="{cy + 28}" text-anchor="middle"
                          fill="{gc}" font-size="32"
                          font-family="DM Serif Display,serif"
                          font-weight="400">{prob_pct:.1f}%</text>
                    <text x="{cx}" y="{cy + 46}" text-anchor="middle"
                          fill="#3D5070" font-size="10"
                          font-family="DM Sans,sans-serif">probabilitas klaim</text>
                </svg>
                <div class="gauge-legend">
                    <span class="gauge-leg-item"><span class="leg-dot" style="background:#34D399;"></span>Rendah (0–40%)</span>
                    <span class="gauge-leg-item"><span class="leg-dot" style="background:#FBBF24;"></span>Sedang (40–70%)</span>
                    <span class="gauge-leg-item"><span class="leg-dot" style="background:#FB7185;"></span>Tinggi (70–100%)</span>
                </div>
            </div>"""
            st.markdown(gauge_html, unsafe_allow_html=True)

            # ── Threshold info ──
            st.markdown(f"""
            <div class="threshold-info">
                <span>📏</span>
                <span>Threshold yang digunakan: <span class="threshold-val">{threshold:.2f}</span></span>
                <span style="margin-left:auto; color:#3D5070; font-size:12px;">
                    {prob:.4f} {'≥' if is_claim else '<'} {threshold:.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Risk factors ──
            risk_factors = []
            safe_factors = []

            if high_risk_destination == 1:
                risk_factors.append(("Destinasi", f"{destination.title()} termasuk destinasi berisiko tinggi"))
            if is_long_trip == 1:
                risk_factors.append(("Durasi", f"Perjalanan panjang {duration_clipped} hari (>14 hari)"))
            if age_clipped >= 60:
                risk_factors.append(("Usia", f"Usia {age_clipped} tahun (kelompok 60+)"))
            if commission_rate >= 0.4:
                risk_factors.append(("Komisi", f"Commission rate tinggi ({commission_rate:.0%})"))
            if agency_type == "Airlines":
                risk_factors.append(("Tipe Agen", "Agen Airlines cenderung lebih berisiko"))

            if high_risk_destination == 0:
                safe_factors.append(("Destinasi", f"{destination.title()} bukan destinasi berisiko tinggi"))
            if is_long_trip == 0:
                safe_factors.append(("Durasi", f"Perjalanan pendek {duration_clipped} hari (≤14 hari)"))
            if age_clipped < 45:
                safe_factors.append(("Usia", f"Usia {age_clipped} tahun (kelompok produktif)"))
            if distribution_channel == "Online":
                safe_factors.append(("Channel", "Distribusi online cenderung lebih rendah risiko"))
            if commission_rate < 0.3:
                safe_factors.append(("Komisi", f"Commission rate wajar ({commission_rate:.0%})"))

            show_risk = risk_factors[:3]
            show_safe = safe_factors[:3]

            factors_html = '<div style="margin-top:16px; padding:16px 18px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:12px;">'
            factors_html += '<div style="color:#3D5070; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.12em; margin-bottom:14px;">🔍 Interpretasi Faktor Risiko</div>'

            if show_risk:
                factors_html += '<div class="risk-section-label rsl-high">↑ Faktor Risiko</div>'
                for name, desc in show_risk:
                    factors_html += (
                        f'<div class="risk-factor-row rf-high">'
                        f'<span class="rf-dot rf-dot-high"></span>'
                        f'<div><div class="rf-name">{name}</div><div class="rf-desc">{desc}</div></div>'
                        f'</div>'
                    )

            if show_safe:
                factors_html += '<div class="risk-section-label rsl-low">↓ Faktor Penurun Risiko</div>'
                for name, desc in show_safe:
                    factors_html += (
                        f'<div class="risk-factor-row rf-low">'
                        f'<span class="rf-dot rf-dot-low"></span>'
                        f'<div><div class="rf-name">{name}</div><div class="rf-desc">{desc}</div></div>'
                        f'</div>'
                    )

            factors_html += '</div>'

            if show_risk or show_safe:
                st.markdown(factors_html, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <strong>⚠️ Error saat melakukan prediksi:</strong><br><br>
                {str(e)}<br><br>
                Pastikan file model kompatibel dengan versi library yang digunakan.
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

# ─── FEATURE ENGINEERING SECTION ───────────────────────────────────────────────
if predict_btn and model_pkg is not None:
    st.markdown("""
    <div class="section-card" style="margin-top: 4px;">
        <div class="section-title">
            <span class="section-title-icon icon-amber">⚙️</span>
            Hasil Feature Engineering
        </div>
    """, unsafe_allow_html=True)

    def bool_badge(val):
        if val == 1:
            return '<span class="fe-badge-yes">Ya (1)</span>'
        return '<span class="fe-badge-no">Tidak (0)</span>'

    fe_items = [
        ("Age (Clipped)",      f"{age_clipped} th"),
        ("Duration (Clipped)", f"{duration_clipped} hari"),
        ("Age Group",          age_group),
        ("Duration Group",     duration_group),
        ("Commission Rate",    f"{commission_rate:.3f}"),
        ("Is Long Trip",       bool_badge(is_long_trip)),
        ("Is Online",          bool_badge(is_online)),
        ("High Risk Dest.",    bool_badge(high_risk_destination)),
    ]
    fe_html = '<div class="fe-grid">'
    for label, value in fe_items:
        fe_html += f"""
        <div class="fe-item">
            <div class="fe-label">{label}</div>
            <div class="fe-value">{value}</div>
        </div>"""
    fe_html += '</div></div>'
    st.markdown(fe_html, unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Travel Insurance Claim Predictor &nbsp;·&nbsp;
    <span>Capstone Project Module 3</span> &nbsp;·&nbsp;
    <span>Logistic Regression Balanced</span>
</div>
""", unsafe_allow_html=True)
