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

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .stApp {
        background: linear-gradient(160deg, #0a1628 0%, #0d2137 50%, #061a2e 100%);
        color: #e8f4fd;
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f35 0%, #0d2540 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #38bdf8;
    }
    [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stNumberInput > div > div > input {
        background: rgba(10, 30, 55, 0.8) !important;
        color: #e8f4fd !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stNumberInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
    }

    /* FIX 1: Hapus padding-top sidebar agar header box tidak terpotong */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0 !important;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #0c2340 0%, #0e3057 40%, #0a3d62 70%, #0b4a7c 100%);
        border-radius: 18px;
        padding: 44px 48px 40px;
        margin-bottom: 32px;
        margin-top: -10px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(56,189,248,0.1) inset;
        position: relative;
        overflow: hidden;
    }
    .header-banner::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 240px; height: 240px;
        background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .header-banner::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(52,211,153,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .header-eyebrow {
        color: #34d399;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .header-eyebrow::before {
        content: '';
        display: inline-block;
        width: 20px; height: 2px;
        background: #34d399;
        border-radius: 2px;
    }
    .header-banner h1 {
        color: #ffffff;
        font-size: 36px;
        font-weight: 900;
        margin: 0 0 10px 0;
        letter-spacing: -1px;
        line-height: 1.15;
        text-shadow: 0 2px 20px rgba(56,189,248,0.3);
    }
    .header-banner h1 .accent {
        background: linear-gradient(90deg, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .header-banner p {
        color: #94a3b8;
        font-size: 15px;
        margin: 0 0 24px 0;
        line-height: 1.6;
        max-width: 520px;
    }
    .header-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .header-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56,189,248,0.1);
        color: #7dd3fc;
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 999px;
        padding: 6px 16px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .header-tag.green {
        background: rgba(52,211,153,0.1);
        color: #6ee7b7;
        border-color: rgba(52,211,153,0.25);
    }
    .header-tag.amber {
        background: rgba(251,191,36,0.1);
        color: #fcd34d;
        border-color: rgba(251,191,36,0.25);
    }

    /* ── Section cards ── */
    .section-card {
        background: rgba(13, 33, 55, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    .section-title {
        color: #38bdf8;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.1);
    }

    /* ── Input summary table ── */
    .input-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }
    .input-item {
        background: rgba(8, 22, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
        transition: border-color 0.2s;
    }
    .input-item:hover {
        border-color: rgba(56, 189, 248, 0.25);
    }
    .input-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .input-value {
        color: #e2e8f0;
        font-size: 14px;
        font-weight: 600;
    }

    /* ── Feature engineering cards ── */
    .fe-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
    }
    .fe-item {
        background: rgba(8, 22, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 10px;
        padding: 14px 12px;
        text-align: center;
    }
    .fe-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .fe-value {
        color: #e2e8f0;
        font-size: 15px;
        font-weight: 700;
    }
    .fe-badge-yes {
        color: #f87171;
        background: rgba(248,113,113,0.12);
        border: 1px solid rgba(248,113,113,0.25);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 600;
    }
    .fe-badge-no {
        color: #34d399;
        background: rgba(52,211,153,0.12);
        border: 1px solid rgba(52,211,153,0.25);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 600;
    }

    /* ── Metric chips ── */
    .metric-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 8px;
    }
    .metric-chip {
        background: rgba(8, 22, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        flex: 1;
        min-width: 90px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .metric-chip:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.08);
    }
    .chip-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .chip-value {
        color: #e2e8f0;
        font-size: 18px;
        font-weight: 800;
    }

    /* ── Threshold info ── */
    .threshold-info {
        background: rgba(8, 22, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 13px;
        color: #94a3b8;
    }
    .threshold-val {
        color: #fbbf24;
        font-weight: 800;
        font-size: 16px;
    }

    /* ── Error box ── */
    .error-box {
        background: rgba(30, 10, 10, 0.8);
        border: 1px solid rgba(248,113,113,0.4);
        border-radius: 12px;
        padding: 22px 26px;
        color: #f87171;
        font-size: 14px;
        margin: 20px 0;
    }

    /* ── Predict button ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0369a1, #0284c7, #0ea5e9) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 0 !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.25) !important;
        margin-top: 8px;
    }
    .stButton > button:hover {
        opacity: 0.92 !important;
        box-shadow: 0 6px 28px rgba(14, 165, 233, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(56, 189, 248, 0.1);
        margin: 22px 0;
    }

    .sidebar-section {
        color: #38bdf8;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 8px 0 6px 0;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15);
        margin-bottom: 14px;
        margin-top: 6px;
    }
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
    <div class="header-eyebrow">Machine Learning · Insurance Analytics</div>
    <h1>✈️ Travel Insurance<br><span class="accent">Claim Predictor</span></h1>
    <p>Platform prediksi klaim polis perjalanan berbasis Machine Learning — dirancang untuk mendukung keputusan bisnis underwriting yang lebih akurat dan efisien.</p>
    <div class="header-tags">
        <span class="header-tag">🤖 Logistic Regression Balanced</span>
        <span class="header-tag green">📊 Binary Classification</span>
        <span class="header-tag amber">⚠️ Imbalanced Data Handling</span>
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
        <pre style="margin-top:10px; color:#e2e8f0; background:rgba(8,22,42,0.8); padding:12px; border-radius:8px;">
├── app.py
├── requirements.txt
└── {MODEL_PATH}
        </pre>
        Jalankan notebook Capstone 3 terlebih dahulu untuk menghasilkan file model, kemudian letakkan file <code>.sav</code> tersebut di folder yang sama dengan <code>app.py</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── SESSION STATE — DEFAULT VALUES ───────────────────────────────────────────
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

# ─── SIDEBAR — INPUT FORM ──────────────────────────────────────────────────────
with st.sidebar:
    # FIX 1: Sidebar header — gunakan st.markdown dengan margin-top: 0 dan pastikan
    # tidak ada elemen Streamlit default di atasnya yang mendorong box ini ke bawah.
    # Kuncinya: hapus st.markdown spacer, dan gunakan CSS padding-top: 0 pada sidebar.
    st.markdown(
        """<div style="text-align:center; padding:20px 10px 16px;
            background:linear-gradient(135deg,rgba(14,165,233,0.14),rgba(52,211,153,0.09));
            border-radius:14px; border:1px solid rgba(56,189,248,0.25);
            margin-bottom:20px; margin-top:1rem;">
            <div style="font-size:38px; line-height:1; margin-bottom:10px;">🛡️</div>
            <div style="color:#e2e8f0; font-size:15px; font-weight:800; letter-spacing:-0.3px; margin-bottom:4px;">
                Input Pemegang Polis</div>
            <div style="color:#94a3b8; font-size:11px; font-weight:500;">Isi semua field di bawah ini</div>
        </div>""",
        unsafe_allow_html=True
    )

    # ── Agency Info ──
    st.markdown('<div class="sidebar-section">🏢 Informasi Agen</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="sidebar-section" style="margin-top:20px;">📦 Produk Asuransi</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="sidebar-section" style="margin-top:20px;">👤 Informasi Pemegang Polis</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="sidebar-section" style="margin-top:20px;">🌍 Informasi Perjalanan</div>', unsafe_allow_html=True)

    destination_options = [
        "SINGAPORE", "MALAYSIA", "THAILAND", "INDONESIA", "AUSTRALIA",
        "JAPAN", "CHINA", "INDIA", "UNITED KINGDOM", "UNITED STATES",
        "HONG KONG", "TAIWAN", "SOUTH KOREA", "PHILIPPINES", "VIETNAM",
        "FRANCE", "GERMANY", "ITALY", "SPAIN", "NETHERLANDS",
        "CANADA", "NEW ZEALAND", "UNITED ARAB EMIRATES", "EGYPT",
        "SOUTH AFRICA", "BRAZIL", "ARGENTINA", "MEXICO", "TURKEY",
        "GREECE", "PORTUGAL", "SWITZERLAND", "AUSTRIA", "SWEDEN",
        "OTHER"
    ]
    destination = st.selectbox("Destination", destination_options,
        index=destination_options.index(st.session_state["destination"]), key="destination")

    duration = st.number_input(
        "Duration (hari)", min_value=0, max_value=730,
        value=st.session_state["duration"], step=1, key="duration",
        help="Durasi perjalanan dalam hari (minimum 0)"
    )

    # ── Financial Info ──
    st.markdown('<div class="sidebar-section" style="margin-top:20px;">💰 Informasi Finansial</div>', unsafe_allow_html=True)

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

    # ── Buttons ──
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Prediksi Klaim", use_container_width=True, type="primary")

    def reset_inputs():
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    st.button("🔄  Reset Input", use_container_width=True, on_click=reset_inputs)

    # FIX 4: Footer sidebar — warna teks lebih terang agar terbaca
    st.markdown("""
    <div style="margin-top:28px; padding:14px 12px; background:rgba(14,36,64,0.85);
                border-radius:10px; border:1px solid rgba(56,189,248,0.2); text-align:center;">
        <div style="color:#94a3b8; font-size:10px; font-weight:700; text-transform:uppercase;
                    letter-spacing:1.5px; margin-bottom:5px;">Capstone Project · Module 3</div>
        <div style="color:#38bdf8; font-size:11px; font-weight:700;">Logistic Regression Balanced</div>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN AREA ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    input_data = {
        "Agency": agency,
        "Agency Type": agency_type,
        "Distribution Channel": distribution_channel,
        "Product Name": product_name,
        "Gender": gender,
        "Age": age,
        "Destination": destination,
        "Duration": f"{duration} hari",
        "Net Sales": f"{net_sales:,.2f}",
        "Commission": f"{commission:,.2f}",
    }

    grid_html = '<div class="input-grid">'
    for label, value in input_data.items():
        grid_html += f"""
        <div class="input-item">
            <div class="input-label">{label}</div>
            <div class="input-value">{value}</div>
        </div>"""
    grid_html += '</div>'

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">📋 Ringkasan Input</div>
        {grid_html}
    </div>
    """, unsafe_allow_html=True)

    recall_color    = "#34d399" if metrics['Recall']  >= 0.7  else ("#fbbf24" if metrics['Recall']  >= 0.5  else "#f87171")
    precision_color = "#64748b"
    f1_color        = "#38bdf8" if metrics['F1']      >= 0.15 else "#fbbf24"
    prauc_color     = "#38bdf8" if metrics['PR AUC']  >= 0.10 else "#fbbf24"
    rocauc_color    = "#38bdf8" if metrics['ROC AUC'] >= 0.70 else "#fbbf24"

    st.markdown(f"""
    <div class="section-card" style="margin-top:0;">
        <div class="section-title">🤖 Informasi Model &amp; Performa</div>
        <div class="metric-row">
            <div class="metric-chip" title="Recall — metrik utama. Mengukur seberapa banyak klaim aktual yang berhasil terdeteksi.">
                <div class="chip-label">⭐ Recall</div>
                <div class="chip-value" style="color:{recall_color};">{metrics['Recall']:.2%}</div>
            </div>
            <div class="metric-chip" title="Precision — rendah adalah wajar pada data imbalanced (~1.7% klaim). Trade-off yang disengaja.">
                <div class="chip-label">Precision ⚠️</div>
                <div class="chip-value" style="color:{precision_color};">{metrics['Precision']:.2%}</div>
            </div>
            <div class="metric-chip" title="F1-Score — harmonic mean antara Precision dan Recall.">
                <div class="chip-label">F1-Score</div>
                <div class="chip-value" style="color:{f1_color};">{metrics['F1']:.2%}</div>
            </div>
            <div class="metric-chip" title="PR AUC — Area Under Precision-Recall Curve. Lebih relevan untuk data sangat imbalanced.">
                <div class="chip-label">PR AUC</div>
                <div class="chip-value" style="color:{prauc_color};">{metrics['PR AUC']:.2%}</div>
            </div>
            <div class="metric-chip" title="ROC AUC — kemampuan model membedakan kelas Claim vs No Claim secara keseluruhan.">
                <div class="chip-label">ROC AUC</div>
                <div class="chip-value" style="color:{rocauc_color};">{metrics['ROC AUC']:.2%}</div>
            </div>
        </div>
        <div style="margin-top:16px; padding:14px 16px; background:rgba(8,22,42,0.6);
                    border:1px solid rgba(251,191,36,0.2); border-left:3px solid #fbbf24;
                    border-radius:0 10px 10px 0; font-size:12.5px; line-height:1.7;">
            <span style="color:#fbbf24; font-weight:800; font-size:13px;">⚠️ Catatan Imbalanced Data</span><br>
            <span style="color:#94a3b8;">
                Dataset klaim sangat tidak seimbang (<strong style="color:#cbd5e1;">~1.7% klaim</strong>).
                Model ini diprioritaskan untuk <strong style="color:#cbd5e1;">Recall tinggi</strong>
                agar klaim aktual tidak terlewat — konsekuensinya Precision menjadi rendah
                (trade-off yang disengaja).
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    if not predict_btn:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🎯 Hasil Prediksi</div>
            <div style="text-align:center; padding: 50px 20px; color:#475569;">
                <div style="font-size:56px; margin-bottom:18px; opacity:0.5;">🔍</div>
                <div style="font-size:16px; font-weight:700; color:#64748b; margin-bottom:8px;">Siap untuk Prediksi</div>
                <div style="font-size:13px; color:#475569; line-height:1.6;">Isi input di sidebar dan klik<br>
                    <strong style="color:#38bdf8; font-size:14px;">Prediksi Klaim</strong> untuk memulai</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
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

        is_long_trip           = 1 if duration_clipped > 14 else 0
        is_online              = 1 if distribution_channel == "Online" else 0
        high_risk_destination  = 1 if destination in high_risk_dest_set else 0

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

            if prob_pct >= 70:
                risk_level  = "VERY HIGH RISK"
                risk_emoji  = "🚨"
                risk_color  = "#f87171"
                risk_bg     = "rgba(248,113,113,0.07)"
                risk_border = "rgba(248,113,113,0.55)"
                risk_desc   = "Risiko sangat tinggi — perlu evaluasi mendalam oleh tim underwriting."
                gauge_color = "#f87171"
            elif prob_pct >= 40:
                risk_level  = "MEDIUM RISK"
                risk_emoji  = "⚠️"
                risk_color  = "#fbbf24"
                risk_bg     = "rgba(251,191,36,0.07)"
                risk_border = "rgba(251,191,36,0.5)"
                risk_desc   = "Risiko sedang — disarankan pemantauan lebih lanjut."
                gauge_color = "#fbbf24"
            elif prob_pct >= threshold * 100:
                risk_level  = "LOW-MEDIUM RISK"
                risk_emoji  = "🔔"
                risk_color  = "#38bdf8"
                risk_bg     = "rgba(56,189,248,0.07)"
                risk_border = "rgba(56,189,248,0.45)"
                risk_desc   = "Risiko di batas threshold — perlu perhatian standar."
                gauge_color = "#38bdf8"
            else:
                risk_level  = "LOW RISK"
                risk_emoji  = "✅"
                risk_color  = "#34d399"
                risk_bg     = "rgba(52,211,153,0.07)"
                risk_border = "rgba(52,211,153,0.45)"
                risk_desc   = "Risiko rendah — pemegang polis kemungkinan tidak mengajukan klaim."
                gauge_color = "#34d399"

            pred_display  = "CLAIM" if is_claim else "NO CLAIM"
            pred_sublabel = "Pemegang polis berpotensi mengajukan klaim." if is_claim else "Pemegang polis kemungkinan tidak mengajukan klaim."
            bar_width     = f"{min(prob_pct, 100):.1f}%"
            thr_left      = f"{threshold * 100:.1f}%"
            thr_label     = f"thr {threshold:.2f}"

            # FIX 3: Progress bar — pisahkan badge risk level ke baris terpisah,
            # beri margin yang cukup antara badge dan progress bar, dan beri
            # height lebih pada threshold marker area agar tidak bertumpuk.
            card_html = (
                f'<div class="section-card" style="padding:20px 22px;">'
                f'<div class="section-title">🎯 Hasil Prediksi</div>'
                f'<div style="background:{risk_bg};border:2px solid {risk_border};border-radius:14px;'
                f'padding:24px 20px 22px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.25);">'
                f'<div style="font-size:52px;margin-bottom:8px;line-height:1;">{risk_emoji}</div>'
                f'<div style="font-size:32px;font-weight:900;color:{risk_color};letter-spacing:-1.5px;margin-bottom:6px;">{pred_display}</div>'
                f'<div style="color:#94a3b8;font-size:13px;margin-bottom:14px;font-weight:500;">{pred_sublabel}</div>'
                # FIX 3a: Risk badge pisah dari progress bar dengan margin bawah yang cukup
                f'<div style="display:inline-block;background:{risk_bg};color:{risk_color};'
                f'border:1.5px solid {risk_border};border-radius:999px;padding:6px 22px;'
                f'font-size:11px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;'
                f'margin-bottom:22px;">{risk_level}</div>'
                # FIX 3b: Progress bar section dengan label dan threshold jelas terpisah
                f'<div style="margin:0 4px 4px 4px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:9px;font-weight:700;'
                f'color:#64748b;margin-bottom:7px;letter-spacing:0.5px;">'
                f'<span>LOW</span><span>MEDIUM</span><span>HIGH</span><span>VERY HIGH</span></div>'
                f'<div style="background:rgba(8,22,42,0.8);border-radius:999px;height:10px;overflow:hidden;">'
                f'<div style="width:{bar_width};height:100%;background:linear-gradient(90deg,#34d399,#fbbf24,#f87171);'
                f'border-radius:999px;"></div></div>'
                # FIX 3c: Threshold marker area — height lebih besar (38px), teks tidak terpotong
                f'<div style="position:relative;height:36px;margin-top:2px;">'
                f'<div style="position:absolute;left:{thr_left};transform:translateX(-50%);'
                f'border-left:2px dashed rgba(251,191,36,0.8);height:14px;top:0;"></div>'
                f'<div style="position:absolute;left:{thr_left};transform:translateX(-50%);'
                f'top:16px;font-size:9.5px;color:#fbbf24;white-space:nowrap;font-weight:800;'
                f'background:rgba(8,22,42,0.85);padding:2px 8px;border-radius:4px;'
                f'border:1px solid rgba(251,191,36,0.25);">{thr_label}</div>'
                f'</div></div>'
                f'<div style="margin-top:10px;padding:11px 14px;background:rgba(8,22,42,0.6);'
                f'border-radius:10px;font-size:12.5px;color:#94a3b8;border-left:3px solid {risk_border};'
                f'text-align:left;line-height:1.5;">{risk_desc}</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            # FIX 2: Gauge chart — perbesar viewBox height agar legend & angka tidak terpotong
            import math

            cx, cy, r = 160, 130, 108

            def polar(angle_deg, radius=r):
                rad = math.radians(angle_deg)
                return cx + radius * math.cos(rad), cy - radius * math.sin(rad)

            def arc_path(start_deg, end_deg, radius=r, inner=76):
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

            zones = [(0,40,"#34d399"),(40,70,"#fbbf24"),(70,100,"#f87171")]
            zone_paths = ""
            for zs, ze, zc in zones:
                zone_paths += f'<path d="{arc_path(p2a(zs), p2a(ze))}" fill="{zc}" opacity="0.22"/>'

            bg_arc = f'<path d="{arc_path(180, 0)}" fill="rgba(8,22,42,0.85)" stroke="rgba(56,189,248,0.12)" stroke-width="0.5"/>'

            needle_angle = p2a(prob_pct)
            if prob_pct > 0:
                active_arc = f'<path d="{arc_path(180, needle_angle)}" fill="{gauge_color}" opacity="0.75"/>'
            else:
                active_arc = ""

            nad = math.radians(needle_angle)
            nx  = cx + 94 * math.cos(nad)
            ny  = cy - 94 * math.sin(nad)

            ticks_svg = ""
            for tv, tl in [(0,"0%"),(25,"25%"),(50,"50%"),(75,"75%"),(100,"100%")]:
                ta = p2a(tv)
                ox,oy = polar(ta, r+5);  ix,iy = polar(ta, r-5);  lx,ly = polar(ta, r+22)
                ticks_svg += f'<line x1="{ix:.1f}" y1="{iy:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" stroke="rgba(100,116,139,0.7)" stroke-width="2"/>'
                ticks_svg += f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" fill="#64748b" font-size="10" font-weight="600">{tl}</text>'

            ta2 = p2a(threshold * 100)
            ox2,oy2 = polar(ta2,r+8); ix2,iy2 = polar(ta2,r-8); lx2,ly2 = polar(ta2,r+28)
            ticks_svg += f'<line x1="{ix2:.1f}" y1="{iy2:.1f}" x2="{ox2:.1f}" y2="{oy2:.1f}" stroke="#fbbf24" stroke-width="2.5" stroke-dasharray="4,2"/>'
            ticks_svg += f'<text x="{lx2:.1f}" y="{ly2:.1f}" text-anchor="middle" fill="#fbbf24" font-size="9.5" font-weight="800">THR</text>'

            for tv, tl, tc in [(20,"LOW","#34d399"),(55,"MED","#fbbf24"),(85,"HIGH","#f87171")]:
                ta = p2a(tv)
                lx, ly = polar(ta, 58)
                ticks_svg += f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" fill="{tc}" font-size="8.5" font-weight="700" opacity="0.8">{tl}</text>'

            # FIX 2: viewBox height 230 → cukup ruang untuk angka + legend di bawah
            # cy turun sedikit (140→130) agar gauge tidak mepet atas
            gauge_html = f"""
            <div class="section-card" style="padding:20px 18px 24px;">
              <div class="section-title">📊 Probabilitas Klaim</div>
              <svg viewBox="0 0 320 215" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:360px; display:block; margin:0 auto; overflow:visible;">
                {bg_arc}
                {zone_paths}
                {active_arc}
                {ticks_svg}
                <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="rgba(0,0,0,0.4)" stroke-width="6" stroke-linecap="round"/>
                <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="{gauge_color}" stroke-width="3.5" stroke-linecap="round"/>
                <circle cx="{cx}" cy="{cy}" r="10" fill="{gauge_color}" opacity="0.9"/>
                <circle cx="{cx}" cy="{cy}" r="6" fill="rgba(8,22,42,0.95)"/>
                <circle cx="{cx}" cy="{cy}" r="3" fill="{gauge_color}"/>
                <text x="{cx}" y="{cy+36}" text-anchor="middle" fill="{gauge_color}" font-size="34" font-weight="900">{prob_pct:.1f}%</text>
                <text x="{cx}" y="{cy+56}" text-anchor="middle" fill="#64748b" font-size="11" font-weight="700" letter-spacing="1.5">PROBABILITAS KLAIM</text>
              </svg>
              <!-- FIX 2: Legend di luar SVG sebagai HTML biasa agar tidak terpotong -->
              <div style="display:flex; justify-content:center; gap:14px; margin-top:12px; flex-wrap:wrap;
                          padding:10px 0 2px; border-top:1px solid rgba(56,189,248,0.08);">
                <span style="font-size:11.5px; color:#34d399; font-weight:600;">🟢 Rendah (0–40%)</span>
                <span style="font-size:11.5px; color:#fbbf24; font-weight:600;">🟡 Sedang (40–70%)</span>
                <span style="font-size:11.5px; color:#f87171; font-weight:600;">🔴 Tinggi (70–100%)</span>
              </div>
            </div>"""
            st.markdown(gauge_html, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="threshold-info">
                <span style="font-size:18px;">📏</span>
                <span style="flex:1;">Threshold yang digunakan: <span class="threshold-val">{threshold:.2f}</span></span>
                <span style="color:#64748b; font-size:13px; font-weight:600;">
                    {prob:.4f} {'≥' if is_claim else '<'} {threshold:.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)

            risk_factors = []
            safe_factors = []

            if high_risk_destination == 1:
                risk_factors.append(("🌍", "Destinasi", f"{destination} termasuk destinasi berisiko tinggi"))
            if is_long_trip == 1:
                risk_factors.append(("📅", "Durasi", f"Perjalanan panjang {duration_clipped} hari (>14 hari)"))
            if age_clipped >= 60:
                risk_factors.append(("👤", "Usia", f"Usia {age_clipped} tahun (kelompok 60+)"))
            if commission_rate >= 0.4:
                risk_factors.append(("💰", "Komisi", f"Commission rate tinggi ({commission_rate:.0%})"))
            if agency_type == "Airlines":
                risk_factors.append(("✈️", "Tipe Agen", "Agen Airlines cenderung lebih berisiko"))
            if duration_group in [">30 hari", "15-30 hari"]:
                risk_factors.append(("🗓️", "Grup Durasi", f"Masuk kategori {duration_group}"))

            if high_risk_destination == 0:
                safe_factors.append(("🌍", "Destinasi", f"{destination} bukan destinasi berisiko tinggi"))
            if is_long_trip == 0:
                safe_factors.append(("📅", "Durasi", f"Perjalanan pendek {duration_clipped} hari (≤14 hari)"))
            if age_clipped < 45:
                safe_factors.append(("👤", "Usia", f"Usia {age_clipped} tahun (kelompok produktif)"))
            if distribution_channel == "Online":
                safe_factors.append(("💻", "Channel", "Distribusi online cenderung lebih rendah risiko"))
            if commission_rate < 0.3:
                safe_factors.append(("💰", "Komisi", f"Commission rate wajar ({commission_rate:.0%})"))

            show_risk = risk_factors[:3]
            show_safe = safe_factors[:3]

            def make_factor_row(icon, label, desc, bg, border_color):
                return (
                    f'<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;'
                    f'padding:10px 12px;background:{bg};border-radius:8px;border-left:2.5px solid {border_color};">'
                    f'<span style="font-size:16px;flex-shrink:0;">{icon}</span>'
                    f'<span style="font-size:12.5px;color:#94a3b8;line-height:1.5;">'
                    f'<strong style="color:#e2e8f0;font-weight:700;">{label}</strong> — {desc}</span>'
                    f'</div>'
                )

            factors_html = ""
            if show_risk:
                factors_html += '<div style="margin-bottom:12px;">'
                factors_html += '<div style="color:#f87171;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">⬆ Faktor Risiko</div>'
                for icon, label, desc in show_risk:
                    factors_html += make_factor_row(icon, label, desc, "rgba(248,113,113,0.07)", "#f87171")
                factors_html += "</div>"

            if show_safe:
                factors_html += "<div>"
                factors_html += '<div style="color:#34d399;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">⬇ Faktor Penurun Risiko</div>'
                for icon, label, desc in show_safe:
                    factors_html += make_factor_row(icon, label, desc, "rgba(52,211,153,0.07)", "#34d399")
                factors_html += "</div>"

            if factors_html:
                st.markdown(
                    '<div class="section-card" style="margin-top:0;">'
                    '<div class="section-title">🔍 Interpretasi Faktor Risiko</div>'
                    + factors_html + "</div>",
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <strong>⚠️ Error saat melakukan prediksi:</strong><br><br>
                {str(e)}<br><br>
                Pastikan file model kompatibel dengan versi library yang digunakan.
            </div>
            """, unsafe_allow_html=True)
            st.stop()

# ─── FEATURE ENGINEERING SECTION ──────────────────────────────────────────────
if predict_btn and model_pkg is not None:
    def bool_badge(val):
        if val == 1:
            return '<span class="fe-badge-yes">Ya (1)</span>'
        return '<span class="fe-badge-no">Tidak (0)</span>'

    fe_items = [
        ("Age (Clipped)", f"{age_clipped} th"),
        ("Duration (Clipped)", f"{duration_clipped} hari"),
        ("Age Group", age_group),
        ("Duration Group", duration_group),
        ("Commission Rate", f"{commission_rate:.3f}"),
        ("Is Long Trip", bool_badge(is_long_trip)),
        ("Is Online", bool_badge(is_online)),
        ("High Risk Dest.", bool_badge(high_risk_destination)),
    ]
    fe_html_inner = '<div class="fe-grid">'
    for label, value in fe_items:
        fe_html_inner += f"""
        <div class="fe-item">
            <div class="fe-label">{label}</div>
            <div class="fe-value">{value}</div>
        </div>"""
    fe_html_inner += '</div>'
    st.markdown(
        '<div class="section-card"><div class="section-title">⚙️ Hasil Feature Engineering</div>'
        + fe_html_inner + '</div>',
        unsafe_allow_html=True
    )

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 28px 0 14px 0;">
    <div style="display:inline-flex; align-items:center; gap:12px; padding:12px 28px;
                background:rgba(8,22,42,0.5); border:1px solid rgba(56,189,248,0.1);
                border-radius:999px;">
        <span style="color:#475569; font-size:12px; font-weight:500;">✈️ Travel Insurance Claim Predictor</span>
        <span style="color:#334155; font-size:10px;">·</span>
        <span style="color:#475569; font-size:12px; font-weight:500;">Capstone Project Module 3</span>
        <span style="color:#334155; font-size:10px;">·</span>
        <span style="color:#475569; font-size:12px; font-weight:500;">Logistic Regression Balanced</span>
    </div>
</div>
""", unsafe_allow_html=True)
