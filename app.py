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
    /* ── Global & Body ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #58a6ff;
    }
    [data-testid="stSidebar"] label {
        color: #c9d1d9 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stNumberInput > div > div > input {
        background: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] .stNumberInput > div > div > input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 14px;
        padding: 36px 40px;
        margin-bottom: 28px;
        border: 1px solid #30363d;
    }
    .header-banner h1 {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: #8b949e;
        font-size: 15px;
        margin: 0;
    }
    .header-tag {
        display: inline-block;
        background: rgba(88,166,255,0.12);
        color: #58a6ff;
        border: 1px solid rgba(88,166,255,0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Section cards ── */
    .section-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 20px;
    }
    .section-title {
        color: #58a6ff;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Input summary table ── */
    .input-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }
    .input-item {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 14px;
    }
    .input-label {
        color: #6e7681;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 3px;
    }
    .input-value {
        color: #e6edf3;
        font-size: 14px;
        font-weight: 500;
    }

    /* ── Feature engineering cards ── */
    .fe-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
    }
    .fe-item {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }
    .fe-label {
        color: #6e7681;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 5px;
    }
    .fe-value {
        color: #e6edf3;
        font-size: 15px;
        font-weight: 600;
    }
    .fe-badge-yes {
        color: #f85149;
        background: rgba(248,81,73,0.1);
        border: 1px solid rgba(248,81,73,0.2);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
    }
    .fe-badge-no {
        color: #3fb950;
        background: rgba(63,185,80,0.1);
        border: 1px solid rgba(63,185,80,0.2);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
    }

    /* ── Result banner ── */
    .result-claim {
        background: linear-gradient(135deg, #1a0a0a, #2a0f0f);
        border: 2px solid #f85149;
        border-radius: 14px;
        padding: 30px 32px;
        text-align: center;
        margin-top: 4px;
    }
    .result-no-claim {
        background: linear-gradient(135deg, #0a1a0a, #0f2a0f);
        border: 2px solid #3fb950;
        border-radius: 14px;
        padding: 30px 32px;
        text-align: center;
        margin-top: 4px;
    }
    .result-icon {
        font-size: 52px;
        margin-bottom: 10px;
    }
    .result-label {
        font-size: 28px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .result-label-claim { color: #f85149; }
    .result-label-no-claim { color: #3fb950; }
    .result-sub {
        color: #8b949e;
        font-size: 14px;
        margin: 0;
    }

    /* ── Risk badge ── */
    .risk-badge-high {
        display: inline-block;
        background: rgba(248,81,73,0.15);
        color: #f85149;
        border: 1px solid rgba(248,81,73,0.35);
        border-radius: 20px;
        padding: 6px 20px;
        font-size: 13px;
        font-weight: 700;
        margin-top: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .risk-badge-low {
        display: inline-block;
        background: rgba(63,185,80,0.12);
        color: #3fb950;
        border: 1px solid rgba(63,185,80,0.25);
        border-radius: 20px;
        padding: 6px 20px;
        font-size: 13px;
        font-weight: 700;
        margin-top: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Probability bar ── */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-top: 6px;
    }
    .prob-bar-bg {
        flex: 1;
        background: #21262d;
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill-high {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #da3633, #f85149);
        transition: width 0.5s ease;
    }
    .prob-bar-fill-low {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #238636, #3fb950);
        transition: width 0.5s ease;
    }
    .prob-pct {
        font-size: 22px;
        font-weight: 800;
        min-width: 60px;
        text-align: right;
    }
    .prob-pct-high { color: #f85149; }
    .prob-pct-low  { color: #3fb950; }

    /* ── Metric chips ── */
    .metric-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 8px;
    }
    .metric-chip {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px 14px;
        text-align: center;
        flex: 1;
        min-width: 80px;
    }
    .chip-label {
        color: #6e7681;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 3px;
    }
    .chip-value {
        color: #e6edf3;
        font-size: 16px;
        font-weight: 700;
    }

    /* ── Threshold info ── */
    .threshold-info {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
        color: #8b949e;
    }
    .threshold-val {
        color: #d29922;
        font-weight: 700;
        font-size: 15px;
    }

    /* ── Error box ── */
    .error-box {
        background: #1a0a0a;
        border: 1px solid #f85149;
        border-radius: 10px;
        padding: 20px 24px;
        color: #f85149;
        font-size: 14px;
        margin: 20px 0;
    }

    /* ── Predict button ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1a4b8c, #2563eb) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 0 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        cursor: pointer !important;
        transition: opacity 0.2s !important;
        margin-top: 8px;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #21262d;
        margin: 20px 0;
    }

    /* ── Sidebar section header ── */
    .sidebar-section {
        color: #58a6ff;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 6px 0 4px 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
MODEL_PATH = "model_travel_insurance_lr_balanced.sav"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model_pkg = load_model(MODEL_PATH)

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>✈️ Travel Insurance Claim Predictor</h1>
    <p>Prediksi potensi klaim pemegang polis berbasis Machine Learning</p>
    <span class="header-tag">🤖 Logistic Regression Balanced</span>
    <span class="header-tag">📊 Binary Classification</span>
    <span class="header-tag">⚠️ Imbalanced Data Handling</span>
</div>
""", unsafe_allow_html=True)

# ─── MODEL ERROR CHECK ─────────────────────────────────────────────────────────
if model_pkg is None:
    st.markdown(f"""
    <div class="error-box">
        <strong>⚠️ File Model Tidak Ditemukan</strong><br><br>
        File <code>{MODEL_PATH}</code> tidak ditemukan di direktori yang sama dengan <code>app.py</code>.<br><br>
        Pastikan struktur direktori Anda seperti berikut:<br>
        <pre style="margin-top:10px; color:#e6edf3; background:#0d1117; padding:10px; border-radius:6px;">
├── app.py
├── requirements.txt
└── {MODEL_PATH}
        </pre>
        Jalankan notebook Capstone 3 terlebih dahulu untuk menghasilkan file model, kemudian letakkan file <code>.sav</code> tersebut di folder yang sama dengan <code>app.py</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

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
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 10px 0;">
        <span style="font-size: 28px;">🛡️</span>
        <div style="color:#e6edf3; font-size:16px; font-weight:700; margin-top:6px;">Input Pemegang Polis</div>
        <div style="color:#6e7681; font-size:12px;">Isi semua field di bawah</div>
    </div>
    <hr style="border-top:1px solid #21262d; margin:10px 0 18px 0;">
    """, unsafe_allow_html=True)

    # ── Agency Info ──
    st.markdown('<div class="sidebar-section">🏢 Informasi Agen</div>', unsafe_allow_html=True)

    agency_options = [
        "EPX", "CWT", "JWT", "RAB", "CBH", "SSI", "KML", "C2B",
        "ADM", "LWC", "TST", "ART", "TTW", "JZI", "CCR", "CSR"
    ]
    agency = st.selectbox("Agency", agency_options, index=0)

    agency_type = st.selectbox(
        "Agency Type",
        ["Airlines", "Travel Agency"],
        index=1
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        ["Online", "Offline"],
        index=0
    )

    # ── Product ──
    st.markdown('<div class="sidebar-section" style="margin-top:14px;">📦 Produk Asuransi</div>', unsafe_allow_html=True)

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
    product_name = st.selectbox("Product Name", product_options, index=0)

    # ── Personal Info ──
    st.markdown('<div class="sidebar-section" style="margin-top:14px;">👤 Informasi Pemegang Polis</div>', unsafe_allow_html=True)

    gender = st.selectbox(
        "Gender",
        ["F", "M", "Unknown"],
        index=2,
        help="Pilih 'Unknown' jika data gender tidak tersedia"
    )

    age = st.number_input(
        "Age (tahun)",
        min_value=0,
        max_value=120,
        value=35,
        step=1,
        help="Usia pemegang polis (akan di-clip ke rentang 18–100)"
    )

    # ── Trip Info ──
    st.markdown('<div class="sidebar-section" style="margin-top:14px;">🌍 Informasi Perjalanan</div>', unsafe_allow_html=True)

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
    destination = st.selectbox("Destination", destination_options, index=0)

    duration = st.number_input(
        "Duration (hari)",
        min_value=0,
        max_value=730,
        value=10,
        step=1,
        help="Durasi perjalanan dalam hari (minimum 0)"
    )

    # ── Financial Info ──
    st.markdown('<div class="sidebar-section" style="margin-top:14px;">💰 Informasi Finansial</div>', unsafe_allow_html=True)

    net_sales = st.number_input(
        "Net Sales (nilai premi)",
        value=50.0,
        step=1.0,
        format="%.2f",
        help="Total nilai penjualan polis / premi"
    )

    commission = st.number_input(
        "Commission in Value",
        min_value=0.0,
        value=12.5,
        step=0.5,
        format="%.2f",
        help="Komisi agen dalam nilai absolut"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Prediksi Klaim", use_container_width=True)

# ─── MAIN AREA ─────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    # ── Input Summary ──
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📋 Ringkasan Input</div>
    """, unsafe_allow_html=True)

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
    st.markdown(grid_html + "</div>", unsafe_allow_html=True)

    # ── Model Info ──
    st.markdown("""
    <div class="section-card" style="margin-top:0;">
        <div class="section-title">🤖 Informasi Model</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-chip">
            <div class="chip-label">Recall</div>
            <div class="chip-value" style="color:#58a6ff;">{metrics['Recall']:.2%}</div>
        </div>
        <div class="metric-chip">
            <div class="chip-label">Precision</div>
            <div class="chip-value" style="color:#58a6ff;">{metrics['Precision']:.2%}</div>
        </div>
        <div class="metric-chip">
            <div class="chip-label">F1-Score</div>
            <div class="chip-value" style="color:#58a6ff;">{metrics['F1']:.2%}</div>
        </div>
        <div class="metric-chip">
            <div class="chip-label">PR AUC</div>
            <div class="chip-value" style="color:#58a6ff;">{metrics['PR AUC']:.2%}</div>
        </div>
        <div class="metric-chip">
            <div class="chip-label">ROC AUC</div>
            <div class="chip-value" style="color:#58a6ff;">{metrics['ROC AUC']:.2%}</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    # ── Prediction Result ──
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🎯 Hasil Prediksi</div>
    """, unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px; color:#6e7681;">
            <div style="font-size:44px; margin-bottom:12px;">🔍</div>
            <div style="font-size:14px;">Isi input di sidebar dan klik<br><strong style="color:#58a6ff;">Prediksi Klaim</strong></div>
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

        is_long_trip           = 1 if duration_clipped > 14 else 0
        is_online              = 1 if distribution_channel == "Online" else 0
        high_risk_destination  = 1 if destination in high_risk_dest_set else 0

        # ── Build input DataFrame ──
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

        # ── Predict ──
        try:
            X_transformed = preprocessor.transform(X_input)
            prob          = float(model.predict_proba(X_transformed)[:, 1][0])
            pred_label    = "Claim" if prob >= threshold else "No Claim"
            is_claim      = pred_label == "Claim"
            prob_pct      = prob * 100

            if is_claim:
                st.markdown(f"""
                <div class="result-claim">
                    <div class="result-icon">⚠️</div>
                    <div class="result-label result-label-claim">{pred_label}</div>
                    <p class="result-sub">Pemegang polis berpotensi mengajukan klaim</p>
                    <span class="risk-badge-high">🔴 HIGH RISK</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-no-claim">
                    <div class="result-icon">✅</div>
                    <div class="result-label result-label-no-claim">{pred_label}</div>
                    <p class="result-sub">Pemegang polis kemungkinan tidak mengajukan klaim</p>
                    <span class="risk-badge-low">🟢 LOW RISK</span>
                </div>
                """, unsafe_allow_html=True)

            # ── Probability bar ──
            fill_class = "prob-bar-fill-high" if is_claim else "prob-bar-fill-low"
            pct_class  = "prob-pct-high" if is_claim else "prob-pct-low"
            st.markdown(f"""
            <div style="margin-top:18px; padding: 16px 18px; background:#0d1117; border:1px solid #21262d; border-radius:10px;">
                <div style="color:#6e7681; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">
                    Probabilitas Klaim
                </div>
                <div class="prob-row">
                    <div class="prob-bar-bg">
                        <div class="{fill_class}" style="width:{min(prob_pct, 100):.1f}%;"></div>
                    </div>
                    <div class="prob-pct {pct_class}">{prob_pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Threshold info ──
            st.markdown(f"""
            <div class="threshold-info">
                <span>📏</span>
                <span>Threshold yang digunakan: <span class="threshold-val">{threshold:.2f}</span></span>
                <span style="margin-left:auto; color:#6e7681;">
                    {prob:.4f} {'≥' if is_claim else '<'} {threshold:.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)

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

# ─── FEATURE ENGINEERING SECTION ──────────────────────────────────────────────
if predict_btn and model_pkg is not None:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">⚙️ Hasil Feature Engineering</div>
    """, unsafe_allow_html=True)

    def bool_badge(val):
        if val == 1:
            return '<span class="fe-badge-yes">Ya (1)</span>'
        return '<span class="fe-badge-no">Tidak (0)</span>'

    fe_html = '<div class="fe-grid">'
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
<div style="text-align:center; padding: 24px 0 10px 0; color:#3d444d; font-size:12px;">
    Travel Insurance Claim Predictor · Capstone Project Module 3 · Logistic Regression Balanced
</div>
""", unsafe_allow_html=True)
