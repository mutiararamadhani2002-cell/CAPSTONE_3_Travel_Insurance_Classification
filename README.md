# 🛡️ Travel Insurance Claim Prediction
*Optimizing Risk Management & Financial Stability in the Travel Industry*

## 📑 1. Introduction & Context
**Travel insurance** merupakan garda terdepan perlindungan perjalanan bagi nasabah. Namun bagi perusahaan, setiap klaim (pembatalan, medis, kecelakaan) adalah biaya operasional yang harus dikelola dengan presisi. 

Proyek ini bertujuan untuk mentransformasi data historis menjadi **predictive insight** guna mengidentifikasi pemegang polis dengan risiko klaim tinggi sebelum kerugian terjadi.


## 🎯 2. The Challenge (Problem Statement)
Perusahaan menghadapi tantangan **Imbalanced Data**:
* **The Gap:** Jumlah nasabah yang mengajukan klaim jauh lebih sedikit dibandingkan yang tidak.
* **The Risk:** Model tradisional sering kali terjebak pada *Accuracy Paradox*—terlihat akurat padahal gagal total dalam mendeteksi nasabah yang benar-benar akan melakukan klaim.


## 🚀 3. Business Objectives
Membangun model klasifikasi cerdas untuk mendukung pilar bisnis utama:
* **Risk Identification:** Mendeteksi profil risiko tinggi secara proaktif.
* **Strategic Pricing:** Mendukung penentuan premi berbasis risiko yang lebih adil.
* **Financial Buffering:** Membantu tim aktuaria dalam menyiapkan pencadangan dana klaim yang lebih akurat.
* **Operational Efficiency:** Mengurangi *unanticipated losses* melalui data-driven decision making.


## 🧪 4. Analytical Approach
Kami menerapkan pendekatan **Supervised Machine Learning (Binary Classification)**:
* **Target:** `Claim` (Yes/No).
* **Features:** Menganalisis korelasi antara *Agency, Product Type, Destination, Duration, Age,* hingga *Net Sales*.
* **Output:** Selain prediksi biner, model memberikan **Risk Score** untuk membantu prioritas evaluasi.


## 👥 5. Key Stakeholders
1.  **Underwriting Team:** Alat bantu verifikasi profil risiko sebelum polis diterbitkan.
2.  **Actuarial & Finance:** Penentuan harga premi dan manajemen cadangan modal.


## ⚠️ 6. Error Impact & Metric Strategy

Dalam industri asuransi, tidak semua kesalahan prediksi bernilai sama:

| Error Type | Reality | Impact |
| :--- | :--- | :--- |
| **False Positive** | Diprediksi klaim, ternyata tidak. | Inefisiensi operasional & biaya evaluasi tambahan. |
| **False Negative** | **Diprediksi aman, ternyata klaim.** | **Bahaya!** Perusahaan tidak siap secara finansial & risiko tidak terantisipasi. |

### **The Strategy: Recall-Focused Evaluation**
Karena **False Negative** membawa risiko kerugian finansial yang lebih besar, model ini memprioritaskan:
-   **Primary Metric:** `Recall` (Memastikan sesedikit mungkin klaim yang lolos dari deteksi).
-   **Secondary Metrics:** `PR AUC` & `F1-Score` untuk menjaga keseimbangan model pada dataset yang tidak seimbang.


## 📈 7. Business Value & Impact
Implementasi model ini bukan sekadar teknis, melainkan langkah strategis untuk:
* ✅ **Visibility:** Memahami profil perilaku nasabah berisiko.
* ✅ **Prevention:** Mitigasi dini terhadap lonjakan klaim yang tak terduga.
* ✅ **Growth:** Meningkatkan profitabilitas melalui manajemen risiko yang lebih tajam.
