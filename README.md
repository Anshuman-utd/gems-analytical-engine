# GeM Tender Intelligence Engine

### AI-powered Tender Analysis, Compliance Checking, Price Prediction & Win Probability for Government e-Marketplace (GeM)

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Groq](https://img.shields.io/badge/Groq-LLM-blueviolet)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063)
![Playwright](https://img.shields.io/badge/Playwright-Web%20Scraping-2EAD33?logo=playwright)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered tender intelligence system that analyzes Government e-Marketplace (GeM) tenders using Large Language Models, Machine Learning, and historical procurement data to assist vendors in making better bidding decisions.

The system automates four key tasks:

- Tender document understanding
- Vendor compliance checking
- Winning (L1) price prediction
- Dynamic win probability estimation



# Features

## Component A — Data Pipeline

- Extracts text from GeM tender PDFs using **pdfplumber**
- Loads and validates historical GeM contract award data
- Cleans duplicate and invalid records
- Provides a unified interface for all downstream components

---

## Component B — AI Compliance Engine

Uses **Groq Llama 3** to convert unstructured tender documents into structured procurement requirements.

Extracts:

- Minimum turnover
- Experience requirement
- EMD requirement
- Delivery period
- MSE preference
- Technical specifications

### Hardened Architecture

The extracted response is validated using **Pydantic**.

If the LLM:

- hallucinates fields
- returns malformed JSON
- violates the schema

the engine automatically falls back to deterministic default values instead of crashing.

Vendor profiles are then evaluated using deterministic Python rules.

---

## Component C — L1 Price Prediction Engine

Predicts the expected winning bid price using historical GeM contract awards.

### Data Sanitization

Before training:

- duplicate removal
- missing value handling
- invalid record removal
- IQR-based outlier detection

This protects the model from:

- predatory pricing
- abnormal quotations
- extreme bidding outliers

### Machine Learning

Model:

- Random Forest Regressor

Feature Engineering:

- Bid duration
- Competition ratio
- Historical bidder statistics
- Organisation information
- Product category
- Ministry
- Turnover requirement
- Quantity

Evaluation Metrics:

- MAE
- RMSE
- R² Score

---

## Component D — Dynamic Win Probability Engine

Estimates a vendor's probability of winning a tender.

Factors considered:

- Proposed bid price
- Predicted L1 price
- Historical competition
- MSE preference
- MII preference

Returns:

- Win Probability (%)
- Strength Verdict
- Supporting metrics

A monotonicity test suite verifies that increasing bid prices always decrease win probability.

---

# Architecture

```text
                    GeM Tender PDF
                           │
                           ▼
                ┌────────────────────┐
                │    Data Pipeline    │
                │ PDF + Historical DB │
                └─────────┬───────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
 ┌────────────────┐              ┌─────────────────┐
 │ Compliance AI  │              │ Pricing Model   │
 │ Groq + Pydantic│              │ Random Forest   │
 └────────────────┘              └─────────────────┘
          │                               │
          └───────────────┬───────────────┘
                          ▼
              Win Probability Engine
                          │
                          ▼
              Tender Intelligence Report
```

---

# Project Structure

```
.
├── scraper.py
├── data_pipeline.py
├── compliance_engine.py
├── pricing_model.py
├── win_probability.py
├── main.py
├── app.py
│
├── data
│   ├── tenders
│   └── bid_results
│
├── requirements.txt
└── README.md
```

---

# Tech Stack

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| LLM | Groq (Llama 3) |
| Validation | Pydantic |
| ML | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| PDF Parsing | pdfplumber |
| Web Scraping | Playwright, BeautifulSoup |
| Data Cleaning | Pandas |
| Logging | Python Logging |

---

# Machine Learning Pipeline

```
Historical GeM Awards
        │
        ▼
Data Cleaning
        │
        ▼
Outlier Removal
        │
        ▼
Feature Engineering
        │
        ▼
Random Forest Training
        │
        ▼
Model Evaluation
        │
        ▼
L1 Price Prediction
```

---

# Compliance Workflow

```
Tender PDF
      │
      ▼
LLM Extraction
      │
      ▼
JSON Output
      │
      ▼
Pydantic Validation
      │
      ▼
Fallback (if invalid)
      │
      ▼
Vendor Evaluation
```

---

# Win Probability Formula

The probability model dynamically considers:

- Vendor Bid
- Predicted Winning Price
- Historical Competition
- Purchase Preference (MSE/MII)

Output:

```
0%  ←───────────────► 100%
Very Weak      Strong
```

---

# Running the Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Configure Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 3. Run the Command Line Interface

```bash
python3 main.py
```

The CLI allows you to:

- Select a tender document
- Select a vendor profile
- Enter a proposed bid price
- View:
  - Compliance evaluation
  - Predicted L1 price
  - Win probability
  - Overall procurement summary

---

## 4. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard provides an interactive interface for:

- Tender selection
- Vendor profile selection
- Bid price analysis
- AI-powered compliance checking
- Machine learning price prediction
- Historical pricing analytics
- Dynamic win probability estimation
- Vendor comparison dashboard

---

# Sample CLI Output

```text
================================================================================
GeM Tender Intelligence Report
================================================================================

Tender
------------------------------------------------------------
File              : GeM-Bidding-9389923.pdf
Quantity          : 15

Product Requirements
------------------------------------------------------------
• No explicit technical specifications extracted.

Compliance Results
------------------------------------------------------------
1. Vendor A — TechSupply India Pvt Ltd ✓
2. Vendor B — GlobalIT Solutions ✗
3. Vendor C — BharatTech Systems ✓  << SELECTED >>

Selected Vendor Analysis
------------------------------------------------------------
Vendor            : Vendor C — BharatTech Systems
Status            : ELIGIBLE ✓

Evaluation Matrix
------------------------------------------------------------
✓ Turnover
✓ Experience
✓ EMD
ℹ MSE Preference (Purchase Preference Applicable)
✓ Technical Specifications
✓ Delivery Timeline

Pricing Model
------------------------------------------------------------
Predicted Winning Price : ₹1,018,159.39

Model Confidence
------------------------------------------------------------
Training Records  : 280
Outliers Removed  : 40
MAE               : ₹126,009
RMSE              : ₹188,674
R² Score          : 0.714

Win Probability
------------------------------------------------------------
Vendor Bid        : ₹1,000,000.00
Predicted L1      : ₹1,018,159.39
Price Difference  : -1.78%
Win Probability   : 78.65%
Verdict           : STRONG

Overall Summary
------------------------------------------------------------
Vendor Status         : ELIGIBLE ✓
Estimated L1 Price    : ₹1,018,159.39
Win Probability       : 78.65%
================================================================================
```

---

# Streamlit Dashboard

The project also includes a web-based Streamlit dashboard that provides an interactive interface for exploring tender intelligence.

Dashboard features include:

- Tender selection
- Vendor profile selection
- Compliance evaluation
- Product requirement extraction
- Historical pricing analytics
- Predicted L1 price
- Win probability estimation
- Vendor comparison
- Executive summary

---

# Model Performance

Current training dataset:

- 360 historical GeM contract awards
- 280 records used after preprocessing
- 40 anomalous bids removed using IQR-based outlier filtering

Model performance:

| Metric | Value |
|---------|-------|
| MAE | ₹126,009 |
| RMSE | ₹188,674 |
| R² Score | 0.714 |

---

# Key Features

- Automated GeM tender ingestion pipeline
- AI-assisted tender requirement extraction using Llama 3 (Groq)
- Structured validation using Pydantic
- Deterministic compliance evaluation engine
- Automatic fallback mechanism for invalid LLM outputs
- Historical GeM pricing analytics
- Outlier-aware machine learning price prediction
- Dynamic vendor win probability estimation
- Interactive Streamlit dashboard
- Modular, extensible project architecture

---

# Project Structure

```
.
├── app.py
├── main.py
├── data_pipeline.py
├── compliance_engine.py
├── pricing_model.py
├── win_probability.py
├── scraper.py
├── vendor_profiles.py
├── data/
│   ├── tenders/
│   └── bid_results/
├── requirements.txt
└── README.md
```

---

# Future Improvements

- Larger historical training dataset for improved prediction accuracy
- OCR support for scanned tender documents
- Real-time GeM API integration
- SHAP-based model explainability
- Vector database for semantic tender search
- Advanced multi-vendor bid strategy recommendations
- Interactive analytics and visualization enhancements

---

# Author

**Anshuman Mehta**

---

# License

This project was developed as part of an AI engineering assignment and is intended for educational and research purposes.