# 💰 AI‑Generated Personal Finance Assistant ML PROJECT (NOTEBOOK)

> **Scope:** This repository focuses on the **data + ML notebook** only (no Streamlit UI). It demonstrates a complete, reproducible pipeline for personal finance analytics, feature engineering, loan‑eligibility prediction, and AI insights via LangChain + OpenAI.

---

## 🔎 Project Overview

This project develops an **AI‑powered personal finance assistant** using **Python, pandas, matplotlib, seaborn, plotly, scikit‑learn, LangChain, and OpenAI**. The assistant:

* Analyzes spending patterns and cashflow
* Engineers financial features (Savings Rate, Debt‑to‑Income Ratio, Monthly Balance, Expense Ratio)
* Predicts **loan eligibility** with a Logistic Regression model
* Produces **LLM‑based insights** and recommendations using **LangChain + OpenAI**
* Operates entirely within a **Jupyter/Colab notebook** for transparency and learning

> The included dataset is **synthetic**. You can swap in any CSV with similar columns.

---

## ✨ Key Capabilities

* **Data Loading & Exploration**: Structured EDA with summary stats & quality checks
* **Feature Engineering**: Rich domain features incl. financial health categories & age bands
* **Visualization**: KPI charts, distributions, correlations, category breakdowns
* **Preprocessing**: NaNs, encoding (Label + One‑Hot), scaling, leakage‑safe splits
* **Loan Prediction**: Logistic Regression baseline (optionally compare XGBoost/LightGBM)
* **Evaluation**: Confusion matrix, classification report, calibrated probability checks
* **Model Persistence**: Save model + encoders with `pickle/joblib`
* **AI Insights (LangChain + OpenAI)**: Prompt templates for (1) general financial advice, (2) loan eligibility rationale, (3) spending analysis

---

## 🧰 Tech Stack (Model Only)

| Layer             | Tools                                                         |
| ----------------- | ------------------------------------------------------------- |
| Language          | Python 3.10+                                                  |
| Data              | pandas, numpy                                                 |
| Viz               | matplotlib, seaborn, plotly                                   |
| ML                | scikit‑learn (LogisticRegression), optional: XGBoost/LightGBM |
| Persistence       | pickle / joblib                                               |
| LLM Orchestration | LangChain                                                     |
| LLM Provider      | OpenAI (GPT‑4 class models)                                   |

---

## 🗃️ Data Schema (Synthetic Example)

| Column            | Type     | Description                                        |
| ----------------- | -------- | -------------------------------------------------- |
| `record_date`     | date     | Transaction date                                   |
| `amount`          | float    | Positive amounts; credit/debit inferred via `type` |
| `type`            | category | `income` / `expense`                               |
| `category`        | category | Groceries, Rent, Transport, Utilities, etc.        |
| `loan_amount`     | float    | Amount of active loan (if any)                     |
| `monthly_income`  | float    | Declared monthly income                            |
| `monthly_expense` | float    | Total monthly expenses                             |
| `age`             | int      | Age in years                                       |
| `gender`          | category | M/F/Other                                          |
| `education_level` | category | High‑School, Bachelor, Master, etc.                |
| `job_title`       | category | Role/occupation                                    |
| `region`          | category | Geographic region                                  |
| `credit_score`    | int      | FICO‑like score                                    |
| `loan_status`     | int      | Target: 1 = Eligible/Approved, 0 = Not Eligible    |

> You can extend with bank‑account, merchant, and repayment history fields.

---

## 🧪 Feature Engineering (What & Why)

**Numeric features**

* **Savings Rate** = `(monthly_income − monthly_expense) / monthly_income` → budget discipline
* **Debt‑to‑Income Ratio (DTI)** = `loan_amount / max(monthly_income, ε)` → lending risk
* **Monthly Balance** = `monthly_income − monthly_expense` → cash cushion
* **Expense Ratio** = `monthly_expense / max(monthly_income, ε)` → affordability
* **Credit Score Range** (binned) → captures non‑linear risk tiers
* **Age Category** (18‑24, 25‑34, 35‑44, 45‑54, 55+) → lifecycle effects

**Temporal features**

* **Month, Quarter, Day of Week, Is Month Start/End** → seasonality & pay‑cycle effects

**Categorical encodings**

* **Label Encoding** for ordinal categories (e.g., education level)
* **One‑Hot** for nominal categories (gender, job\_title, region, category)

**Targets & helper labels**

* **Financial Health** category derived from Savings Rate & DTI (e.g., `Good`, `Watch`, `Risk`)

> These features improve both **predictive power** and **explainability** for user‑facing insights.

---

## 📈 EDA & Visualization (Notebook Sections)

* Distribution plots: income, expenses, DTI, savings rate
* Category spend breakdowns (stacked bar / treemap)
* Credit score vs. DTI (scatter with LOWESS trend)
* Loan status by gender/loan type (countplots)
* Average credit score by region (bar)
* Age‑group distribution by gender (heatmap)
* Correlation heatmap on encoded & scaled matrix

---

## 🧹 Preprocessing Pipeline

1. **Missing Values**: impute (mode for categorical, median for numeric)
2. **Encoding**: LabelEncoder (ordinal) + One‑Hot (nominal)
3. **Scaling**: StandardScaler for numeric features
4. **Split**: `train_test_split` (stratify by `loan_status`), seed=42
5. **Leakage Guard**: Fit transformers **only on train**; transform test separately

> The notebook shows a clean, reproducible `ColumnTransformer` + `Pipeline` setup.

---

## 🤖 Loan Eligibility Model

* **Baseline**: `LogisticRegression` (class\_weight="balanced", max\_iter=1000)
* **Metrics**: Accuracy, Precision/Recall/F1 (macro), ROC‑AUC; confusion matrix
* **Calibration**: Optional `CalibratedClassifierCV` for probability quality
* **Model persistence**: `pickle`/`joblib` for model + preprocessors

**Why Logistic Regression?**

* Interpretable coefficients and odds ratios
* Strong baseline; fast to train; robust with regularization

> Notebook includes optional comparison against tree‑based models for robustness.

---

## 🧠 AI Insights with LangChain + OpenAI (Notebook‑only)

* **General Financial Insight Prompt** — summarises user’s financial posture
* **Loan Eligibility Rationale Prompt** — explains approval/decline drivers
* **Spending Analysis Prompt** — suggests actionable savings targets

**Prompt Template (example)**

```
You are a professional financial coach. Given:
- User snapshot: {user_json}
- Model prediction: {prediction_json}
Explain the top 3 strengths and top 3 risks in 120 words max.
Provide 3 specific actions for the next 30–90 days.
Use simple language and avoid jargon.
```

> The notebook demonstrates secure key loading and graceful fallback when the API/key isn’t available.

---

## 🧱 Architecture (Model‑Only View)

```
┌────────────────────┐    ┌───────────────────────┐    ┌──────────────────────┐
│   CSV / Synthetic  │ →  │  Data Validation &    │ →  │  Feature Engineering  │
│     Dataset        │    │   EDA (pandas/plots)  │    │  (ratios, bins, time)│
└────────────────────┘    └───────────────────────┘    └──────────────────────┘
                                      │
                                      ▼
                         ┌───────────────────────────┐
                         │  Preprocess Pipeline      │
                         │ (encode + scale + split)  │
                         └───────────────────────────┘
                                      │
                                      ▼
                         ┌───────────────────────────┐
                         │  Logistic Regression      │
                         │  (train/evaluate/save)    │
                         └───────────────────────────┘
                                      │
                                      ▼
                         ┌───────────────────────────┐
                         │  LangChain + OpenAI       │
                         │  (insights & advice)      │
                         └───────────────────────────┘
```

---

## 🗺️ Roadmap (Milestones)

**M0 – Data & Notebook (this repo)**

* ✅ Synthetic CSV + schema
* ✅ EDA + visualizations
* ✅ Feature engineering
* ✅ Logistic Regression baseline + evaluation
* ✅ Model persistence
* ✅ LangChain + OpenAI insights (optional)

**M1 – Robustness**

* Cross‑validation, hyperparameter search
* Class‑imbalance strategies (SMOTE, thresholds)
* Add XGBoost/LightGBM benchmarks
* SHAP/Permutation importance for explainability

**M2 – Productionization (future)**

* API service (FastAPI) wrapping model
* Batch/stream scoring, monitoring & drift checks
* Data contracts and Pydantic validation

**M3 – Full App (future, not in this repo)**

* Streamlit/Frontend UI
* Secure auth, secrets, and role‑based access
* Real bank connectors (Plaid, etc.)

---

## 🧑‍💻 Getting Started (Colab & Local)

### Option A — Google Colab

1. Upload the notebook and dataset CSV
2. Run the **Setup** cell to install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn langchain openai
   ```
3. (Optional) Add **OpenAI** key via Colab **Secrets** as `OPENAI_API_KEY`
4. Run cells sequentially: EDA → Features → Preprocess → Train/Eval → Persist → LLM Insights

### Option B — Local (VS Code/Jupyter)

1. Clone the repo & create a virtual environment

   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Put your CSV in `data/`
3. Launch Jupyter and open the notebook

   ```bash
   jupyter lab  # or: jupyter notebook
   ```
4. (Optional) `export OPENAI_API_KEY=...` to enable LLM insights

---

## ✅ Expected Results (Illustrative)

* **Model**: Logistic Regression with balanced classes
* **Metrics**: Macro‑F1 ≥ 0.80 on synthetic test set (varies by seed)
* **Artifacts**: `model.pkl`, `preprocess.pkl`, and `metrics.json`
* **Outputs**: Plots (distributions, correlations), confusion matrix, LLM text insights

> Replace the synthetic CSV with your data to get personalized metrics.

---

## 🔐 Security & Ethics

* Do not upload real personal finance data to public notebooks
* Anonymize PII (names, account numbers, addresses)
* Clearly label synthetic vs. real data
* Provide disclaimers: **Not financial advice**; consult a licensed advisor

---

## 🧩 Repo Structure (Suggested)

```
📦 ai-finance-assistant-model
├─ 📂 data/                     # synthetic or user CSVs
├─ 📂 notebooks/
│  └─ finance_model.ipynb       # main notebook (this project)
├─ 📂 artifacts/
│  ├─ model.pkl                 # trained model
│  └─ preprocess.pkl            # encoders/scalers
├─ requirements.txt
└─ README.md
```

---

## 🧷 Troubleshooting

* **Shapes/columns don’t match at predict time** → Save & reuse the same `ColumnTransformer` that was fit on train
* **Poor recall on minority class** → Use `class_weight='balanced'`, threshold tuning, or rebalancing
* **OpenAI errors** → Ensure `OPENAI_API_KEY` is set; add try/except around API calls
* **Synthetic data too easy/hard** → Adjust noise, class balance, and feature overlap

---

## 🙌 Contributing

PRs welcome! Please open an issue to discuss feature requests, datasets, or evaluation protocols.

---

## 🧾 License

MIT License — see `LICENSE` for details.

---

### 📣 Summary (Executive)

A clean, extensible **model‑only** pipeline for an AI personal finance assistant, from **EDA → Features → ML → Evaluation → Persistence → LLM Insights**. It’s recruiter‑friendly, instructional, and ready to be extended into a full application when you’re ready.
