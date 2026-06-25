# Two Decades of IPO Performance
### Analyzing 2,237 US IPOs (2004–2024) using Python, KMeans Clustering & Random Forest

> **Built entirely in Google Colab · 100% free data · Fully reproducible**

---

## What This Project Does

We used Python and machine learning to analyze every US IPO listed between 2004 and 2024 — 2,237 companies in total — and measured how each one performed against the S&P 500 using Buy-and-Hold Abnormal Return (BHAR).

The goal: find patterns in IPO performance that are invisible to the naked eye but detectable through data.

---

## Key Findings

| Finding | Number |
|---|---|
| Total IPOs analyzed | 2,237 |
| Study period | 2004 – 2024 |
| Benchmark | S&P 500 (^GSPC) |
| IPOs still underperforming the index today | 72% |
| Best cohort (2012) — mean 1-yr BHAR | +18% vs S&P 500 |
| Worst cohort (2021) — mean 1-yr BHAR | −35% vs S&P 500 |
| IPOs in the 2021 cohort alone | 1,035 |
| Alpha Compounders median BHAR today | +1,711% |
| Model F1 to detect Alpha Compounders at listing | 0.066 (near random) |
| Listing-Day Reversals F1 at listing | 1.000 (perfectly identifiable) |

---

## The Four IPO Archetypes (KMeans, k=4)

Machine learning identified four natural performance archetypes from the return trajectory data — not predefined categories, but patterns the algorithm found on its own.

| Archetype | n | Median Listing Return | Median 1-yr BHAR | Median BHAR Today |
|---|---|---|---|---|
| 🟢 Alpha Compounders | 52 | −0.6% | +22.1% | +2,047% |
| 🟡 Gradual Outperformers | 553 | +1.2% | +55.8% | +626% |
| 🔴 Listing-Day Reversals | 19 | +134.2% | −71.6% | −178% |
| ⚫ Chronic Underperformers | 1,613 | −0.8% | −39.5% | −180% |

**The finding that surprised us most:**

Alpha Compounders — the companies generating 2,000%+ lifetime returns — are statistically indistinguishable from Chronic Underperformers on listing day (F1 = 0.066). The future Googles looked exactly like the future Groupons when they listed. The algorithm couldn't separate them. Neither could the market.

---

## The Bubble Chart

Each bubble = one year of IPOs. Size = number of companies listed. Colour = mean 1-year BHAR vs S&P 500.

The pattern is consistent across 20 years: **the more companies rushed to list, the worse the average outcome.**

The market didn't get better at pricing new companies. It got better at distributing overpriced ones.

---

## Methodology

### 1. Data Collection
- **IPO universe**: NASDAQ Screener API — all US operating companies listing 2004–2024
- **Price history**: `yfinance` — daily OHLCV from listing date to June 2026
- **Benchmark**: S&P 500 total return index (`^GSPC`)
- **Market volatility**: CBOE VIX (`^VIX`) via yfinance
- **Cost**: $0 — no paid data sources used

### 2. BHAR Calculation
```
BHAR(T) = [(P₁ − P₀) / P₀] − [(S₁ − S₀) / S₀]
```
Where P₀/P₁ = IPO stock price at listing/horizon T, S₀/S₁ = S&P 500 at same dates.

Calculated at: 30 days · 90 days · 1 year · 3 years · current date

### 3. KMeans Clustering
- Features: `listing_return`, `bhar_1yr`, `bhar_today`
- Preprocessing: StandardScaler (z-score normalisation)
- Outlier treatment: winsorised at 1st and 99th percentiles
- Optimal k: determined via elbow method + silhouette score → k = 4

### 4. Random Forest Classifier
- Target: KMeans archetype (4 classes)
- Features (listing-day only): `listing_return`, `vix_at_ipo`, `ipo_year`, `sector`, `era`
- Validation: 5-fold stratified cross-validation
- Accuracy: 60.1% (listing-day signals) → 93.2% (adding 1-year BHAR)
- Class weights: balanced (handles class imbalance)

### 5. OLS Regression
- Dependent variable: BHAR today
- Independent variables: era dummies, listing return, VIX at listing
- Baseline: 2012–2019 long-bull era
- Cohort-level R²: 0.556 (annual means) — IPO vintage explains 55.6% of cohort performance variation
- Individual-level R²: 0.08 — individual outcomes remain largely unpredictable

---

## Era Analysis

| Era | Years | % Chronic Underperformers | % Alpha Compounders |
|---|---|---|---|
| Pre-GFC Bull | 2004–2007 | 62% | 9% |
| Crisis & Recovery | 2008–2011 | 73% | 5% |
| Long Bull Run | 2012–2019 | 67% | 2% |
| ZIRP Bubble | 2020–2021 | **81%** | 0% |
| Rate Hike Cycle | 2022–2024 | 79% | 1% |

---

## Repository Structure

```
ipo-performance-analysis/
│
├── README.md
│
├── notebooks/
│   ├── 01_data_collection.ipynb       # NASDAQ API + yfinance pipeline
│   ├── 02_bhar_calculation.ipynb      # BHAR at all time horizons
│   ├── 03_kmeans_clustering.ipynb     # Elbow method + KMeans archetypes
│   └── 04_random_forest.ipynb        # Classification + feature importance
│
├── charts/
│   ├── bubble_chart.png               # Mean BHAR by IPO year
│   ├── trajectory_lines.png           # Return path by archetype
│   ├── archetype_heatmap.png          # % archetype by era
│   ├── feature_importance.png         # Random forest feature importance
│   └── regression_era.png            # OLS era coefficients
│
├── data/
│   └── ipo_master.csv                 # 2,237 IPOs with all features
│
└── requirements.txt
```

---

## Requirements

```
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
scipy>=1.11.0
requests>=2.31.0
tqdm>=4.65.0
pytz>=2023.3
```

Install with:
```bash
pip install -r requirements.txt
```

---

## How to Run

1. Clone this repo
2. Open any notebook in [Google Colab](https://colab.research.google.com)
3. Run cells in order — all data is fetched automatically, no downloads needed
4. Full pipeline takes approximately 45 minutes for the complete 2,237-ticker fetch

```bash
git clone https://github.com/yourusername/ipo-performance-analysis
```

---

## Limitations

- NASDAQ screener captures currently-listed or recently-delisted tickers only — companies delisted more than several years ago may be under-represented (survivorship bias caveat)
- BHAR assumes continuous holding from listing — not a tradeable strategy
- KMeans assigns hard cluster boundaries — real performance is a continuum
- Individual-level R² of 0.08 confirms individual IPO outcomes are inherently unpredictable — cohort and era-level findings are more robust

---

## Citation

If you use this work, please cite:

```
IPO Performance Analysis: A Machine Learning Classification of 2,237 US IPOs (2004–2024)
GitHub: github.com/yourusername/ipo-performance-analysis
June 2026
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Built with Python · scikit-learn · yfinance · Google Colab · Zero paid data*
