# ============================================================
# US IPO RESEARCH — COMPLETE NOTEBOOK
# Run each cell block in Google Colab in order
# 2,237 US IPOs · 2004–2024 · KMeans + Random Forest
# ============================================================


# ────────────────────────────────────────────────────────────
# CELL 1 — Install libraries
# ────────────────────────────────────────────────────────────
"""
!pip install yfinance pandas numpy requests scikit-learn matplotlib scipy tqdm pytz -q
"""


# ────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ────────────────────────────────────────────────────────────
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
import pytz
from datetime import date, timedelta
from tqdm.notebook import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
NY = pytz.timezone("America/New_York")
print("✓ All libraries loaded")
"""


# ────────────────────────────────────────────────────────────
# CELL 3 — Fetch NASDAQ ticker list (free API)
# ────────────────────────────────────────────────────────────
"""
def fetch_nasdaq_tickers():
    url = ("https://api.nasdaq.com/api/screener/stocks"
           "?tableonly=true&limit=25000&offset=0&download=true")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept":     "application/json, text/plain, */*",
        "Referer":    "https://www.nasdaq.com/",
    }
    r = requests.get(url, headers=headers, timeout=20)
    data = r.json()
    return pd.DataFrame(data['data']['rows'])

print("Fetching NASDAQ screener...")
nasdaq_raw = fetch_nasdaq_tickers()
print(f"Total tickers: {len(nasdaq_raw)}")
print(f"Columns: {nasdaq_raw.columns.tolist()}")
"""


# ────────────────────────────────────────────────────────────
# CELL 4 — Filter to IPO years 2004–2024
# ────────────────────────────────────────────────────────────
"""
nasdaq = nasdaq_raw.rename(columns={
    'symbol':   'ticker',
    'name':     'company',
    'ipoyear':  'ipo_year',
    'sector':   'sector',
    'industry': 'industry',
    'marketCap':'market_cap',
    'lastsale': 'current_price',
})

nasdaq['ipo_year'] = pd.to_numeric(nasdaq['ipo_year'], errors='coerce')

ipo_universe = nasdaq[
    (nasdaq['ipo_year'] >= 2004) &
    (nasdaq['ipo_year'] <= 2024)
].copy()

# Remove non-stock tickers
ipo_universe = ipo_universe[ipo_universe['ticker'].str.len() <= 5]
ipo_universe = ipo_universe[
    ~ipo_universe['ticker'].str.contains(
        r'[\^/\.\+\-W]', regex=True, na=False)
]
ipo_universe = ipo_universe[
    ipo_universe['sector'].notna() &
    (ipo_universe['sector'] != '')
]

tickers_list = ipo_universe['ticker'].dropna().unique().tolist()
print(f"IPOs 2004–2024 after filtering: {len(ipo_universe)}")
print(f"\nBy year:\n{ipo_universe.groupby('ipo_year').size().to_string()}")
"""


# ────────────────────────────────────────────────────────────
# CELL 5 — Helper functions
# ────────────────────────────────────────────────────────────
"""
def get_ipo_data(ticker):
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="max")
        if hist.empty:
            return {'ticker': ticker, 'dead': True,
                    'ipo_date': None, 'ipo_close': None,
                    'current_price': None}
        ipo_date  = hist.index[0].date()
        ipo_close = round(hist['Close'].iloc[0], 2)
        curr      = round(hist['Close'].iloc[-1], 2)
        return {'ticker': ticker, 'dead': False,
                'ipo_date': ipo_date, 'ipo_close': ipo_close,
                'current_price': curr}
    except Exception:
        return {'ticker': ticker, 'dead': True,
                'ipo_date': None, 'ipo_close': None,
                'current_price': None}


def compute_returns(ticker, ipo_date):
    try:
        t    = yf.Ticker(ticker)
        sp   = yf.Ticker("^GSPC")
        hist = t.history(start=str(ipo_date), end="2026-06-14")
        sp_h = sp.history(start=str(ipo_date), end="2026-06-14")

        if hist.empty or sp_h.empty:
            return {}

        p0  = hist['Close'].iloc[0]
        sp0 = sp_h['Close'].iloc[0]

        def bhar(days=None, use_last=False):
            if use_last:
                p1 = hist['Close'].iloc[-1]
                s1 = sp_h['Close'].iloc[-1]
            else:
                td           = pd.Timestamp(str(ipo_date), tz=NY) + pd.Timedelta(days=days)
                future_stock = hist.loc[hist.index >= td, 'Close']
                future_sp    = sp_h.loc[sp_h.index >= td, 'Close']
                if future_stock.empty or future_sp.empty:
                    return None
                p1 = future_stock.iloc[0]
                s1 = future_sp.iloc[0]
            return round(((p1 - p0) / p0 - (s1 - sp0) / sp0) * 100, 2)

        return {
            'listing_return': round(
                (hist['Close'].iloc[0] - hist['Open'].iloc[0])
                / hist['Open'].iloc[0] * 100, 2),
            'bhar_30d':   bhar(30),
            'bhar_90d':   bhar(90),
            'bhar_1yr':   bhar(365),
            'bhar_3yr':   bhar(3 * 365),
            'bhar_today': bhar(use_last=True),
        }
    except Exception as e:
        print(f"    error {ticker}: {e}")
        return {}


# VIX via yfinance
vix_raw  = yf.Ticker("^VIX")
vix_hist = vix_raw.history(start="2004-01-01", end="2026-06-14")
vix_df   = vix_hist[['Close']].reset_index()
vix_df.columns = ['date', 'vix']
vix_df['date'] = vix_df['date'].dt.tz_localize(None)
print(f"✓ VIX loaded: {vix_df['date'].min().date()} → {vix_df['date'].max().date()}")

def get_vix_on_date(ipo_date):
    future = vix_df[vix_df['date'].dt.date >= ipo_date]
    if future.empty:
        return None
    return round(future['vix'].iloc[0], 1)

def classify_regime(vix):
    if vix is None or pd.isnull(vix): return 'unknown'
    if vix < 15:  return 'calm'
    if vix < 25:  return 'normal'
    return 'fearful'

def classify_era(ipo_date):
    if ipo_date is None: return 'unknown'
    y = ipo_date.year
    if 2004 <= y <= 2007: return 'pre_GFC_bull'
    if 2008 <= y <= 2011: return 'crisis_recovery'
    if 2012 <= y <= 2019: return 'long_bull'
    if 2020 <= y <= 2021: return 'zirp_bubble'
    if 2022 <= y <= 2024: return 'rate_hike'
    return 'other'

print("✓ All helper functions defined")
"""


# ────────────────────────────────────────────────────────────
# CELL 6 — Quick test with 21 known IPOs (run this first)
# ────────────────────────────────────────────────────────────
"""
SAMPLE_TICKERS = [
    'GOOG', 'V', 'GRPN', 'META', 'BABA', 'SHOP',
    'UBER', 'LYFT', 'PINS', 'ZM',
    'SNOW', 'ABNB', 'DASH',
    'RIVN', 'COIN', 'RBLX', 'HOOD',
    'ARM', 'CART', 'RDDT',
]

sample_results = []
print(f"Quick test: {len(SAMPLE_TICKERS)} known IPOs\n")

for tk in SAMPLE_TICKERS:
    row = get_ipo_data(tk)
    if not row['dead'] and row['ipo_date']:
        ret = compute_returns(tk, row['ipo_date'])
        row.update(ret)
        vix = get_vix_on_date(row['ipo_date'])
        row['vix_at_ipo']    = vix
        row['market_regime'] = classify_regime(vix)
        row['era']           = classify_era(row['ipo_date'])
    sample_results.append(row)
    print(f"  {tk}: bhar_1yr={row.get('bhar_1yr')}% | bhar_today={row.get('bhar_today')}%")
    time.sleep(0.5)

sample_df = pd.DataFrame(sample_results)
print(f"\n✓ Done: {len(sample_df)} rows")
sample_df[['ticker','ipo_date','bhar_1yr','bhar_3yr','bhar_today','era','market_regime']]
"""


# ────────────────────────────────────────────────────────────
# CELL 7 — Full fetch loop (run after test passes)
# Takes ~30–45 mins for full dataset
# ────────────────────────────────────────────────────────────
"""
all_results = []
errors      = []

for tk in tqdm(tickers_list, desc="Fetching IPO returns"):
    row = get_ipo_data(tk)
    if not row['dead'] and row['ipo_date']:
        ret = compute_returns(tk, row['ipo_date'])
        row.update(ret)
        vix = get_vix_on_date(row['ipo_date'])
        row['vix_at_ipo']    = vix
        row['market_regime'] = classify_regime(vix)
        row['era']           = classify_era(row['ipo_date'])
    else:
        errors.append(tk)
    all_results.append(row)
    time.sleep(0.25)

results_df = pd.DataFrame(all_results)
print(f"\n✓ Done")
print(f"Alive  : {(~results_df['dead']).sum()}")
print(f"Dead   : {results_df['dead'].sum()}")
print(f"Errors : {len(errors)}")
"""


# ────────────────────────────────────────────────────────────
# CELL 8 — Build master DataFrame
# ────────────────────────────────────────────────────────────
"""
meta_cols = ['ticker','company','ipo_year','sector','industry']

master = ipo_universe[meta_cols].merge(
    results_df, on='ticker', how='inner'
)

master['still_alive'] = ~master['dead'].fillna(True)

print(f"Master dataset: {len(master)} rows × {len(master.columns)} cols")
print(f"Alive: {master['still_alive'].sum()} | Dead: {(~master['still_alive']).sum()}")
print(f"\nMean BHAR by era:")
print(master.groupby('era')[['bhar_1yr','bhar_3yr','bhar_today']].mean().round(1).to_string())

master.to_csv('ipo_master.csv', index=False)
print("\n✓ Saved: ipo_master.csv")
"""


# ────────────────────────────────────────────────────────────
# CELL 9 — KMeans clustering
# ────────────────────────────────────────────────────────────
"""
cluster_features = ['listing_return', 'bhar_1yr', 'bhar_today']

cluster_df = master[cluster_features + ['ticker','company','sector','era','ipo_year']].dropna()
print(f"Rows for clustering: {len(cluster_df)}")

# Remove outliers (1st–99th pct)
for col in cluster_features:
    p01 = cluster_df[col].quantile(0.01)
    p99 = cluster_df[col].quantile(0.99)
    cluster_df = cluster_df[(cluster_df[col] >= p01) & (cluster_df[col] <= p99)]

print(f"After outlier removal: {len(cluster_df)}")

X        = cluster_df[cluster_features].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
"""


# ────────────────────────────────────────────────────────────
# CELL 10 — Elbow method
# ────────────────────────────────────────────────────────────
"""
inertias    = []
silhouettes = []
K_range     = range(2, 11)

print("Running elbow method...")
for k in K_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, lbl)
    silhouettes.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.1f} | silhouette={sil:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(list(K_range), inertias, 'o-', color='#534AB7', linewidth=2, markersize=7)
ax1.set_xlabel('Number of clusters (k)', fontsize=11)
ax1.set_ylabel('Inertia (WCSS)', fontsize=11)
ax1.set_title('Elbow Method', fontsize=12, fontweight='bold')
ax1.set_xticks(list(K_range))
ax1.grid(True, alpha=0.3)

best_k = list(K_range)[silhouettes.index(max(silhouettes))]
ax1.axvline(x=4, color='#E24B4A', linestyle='--', linewidth=1.5, label='k=4 selected')
ax1.legend()

ax2.plot(list(K_range), silhouettes, 's-', color='#1D9E75', linewidth=2, markersize=7)
ax2.set_xlabel('Number of clusters (k)', fontsize=11)
ax2.set_ylabel('Silhouette Score', fontsize=11)
ax2.set_title('Silhouette Score', fontsize=12, fontweight='bold')
ax2.set_xticks(list(K_range))
ax2.grid(True, alpha=0.3)
ax2.axvline(x=4, color='#E24B4A', linestyle='--', linewidth=1.5, label='k=4 selected')
ax2.legend()

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Elbow chart saved")
"""


# ────────────────────────────────────────────────────────────
# CELL 11 — Fit KMeans k=4 and name archetypes
# ────────────────────────────────────────────────────────────
"""
K_FINAL  = 4
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
cluster_df = cluster_df.copy()
cluster_df['cluster'] = km_final.fit_predict(X_scaled)

print("Cluster means:")
print(cluster_df.groupby('cluster')[cluster_features].mean().round(1).to_string())
print(f"\nCluster sizes:")
print(cluster_df['cluster'].value_counts().sort_index().to_string())

# ── Name clusters by bhar_today rank ──
means  = cluster_df.groupby('cluster')[cluster_features].mean()
ranked = means['bhar_today'].rank()

archetype_map = {}
for c in means.index:
    r = ranked[c]
    if r == K_FINAL:       archetype_map[c] = 'Alpha Compounders'
    elif r == K_FINAL - 1: archetype_map[c] = 'Gradual Outperformers'
    elif r == 2:           archetype_map[c] = 'Listing-Day Reversals'
    else:                  archetype_map[c] = 'Chronic Underperformers'

cluster_df['archetype'] = cluster_df['cluster'].map(archetype_map)

print(f"\nArchetype distribution:")
print(cluster_df['archetype'].value_counts().to_string())
"""


# ────────────────────────────────────────────────────────────
# CELL 12 — Merge archetypes into master
# ────────────────────────────────────────────────────────────
"""
cluster_labels = cluster_df[['ticker','cluster','archetype']].copy()
master_full    = master.merge(cluster_labels, on='ticker', how='left')

cluster_rich = master_full[master_full['archetype'].notna()].copy()

print(f"cluster_rich: {len(cluster_rich)} rows")
print(cluster_rich['archetype'].value_counts().to_string())

colors = {
    'Alpha Compounders':      '#1D9E75',
    'Gradual Outperformers':  '#EF9F27',
    'Listing-Day Reversals':  '#E24B4A',
    'Chronic Underperformers':'#4A6FA5',
}
"""


# ────────────────────────────────────────────────────────────
# CELL 13 — Trajectory line chart
# ────────────────────────────────────────────────────────────
"""
all_return_cols = ['listing_return','bhar_30d','bhar_90d',
                   'bhar_1yr','bhar_3yr','bhar_today']
horizons = [c for c in all_return_cols if c in cluster_rich.columns]
horizon_labels = {
    'listing_return': 'Listing Day',
    'bhar_30d':  '30 Days',
    'bhar_90d':  '90 Days',
    'bhar_1yr':  '1 Year',
    'bhar_3yr':  '3 Years',
    'bhar_today':'Current',
}
arch_order = ['Alpha Compounders','Gradual Outperformers',
              'Listing-Day Reversals','Chronic Underperformers']

fig, ax = plt.subplots(figsize=(12, 6))
for arch in arch_order:
    grp     = cluster_rich[cluster_rich['archetype'] == arch]
    medians = [grp[h].median() for h in horizons]
    labels  = [horizon_labels[h] for h in horizons]
    ax.plot(labels, medians, marker='o', linewidth=2.2,
            markersize=7, color=colors[arch],
            label=f"{arch} (n={len(grp)})")
    ax.annotate(f"{medians[-1]:.0f}%",
                xy=(len(horizons)-1, medians[-1]),
                xytext=(len(horizons)-0.85, medians[-1]),
                fontsize=9, color=colors[arch], va='center')

ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel('Time Horizon', fontsize=11)
ax.set_ylabel('Median BHAR vs S&P 500 (%)', fontsize=11)
ax.set_title('IPO Performance Trajectory by Classification\n'
             'Median Buy-and-Hold Abnormal Return (BHAR) vs S&P 500',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('trajectory_final.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Trajectory saved")
"""


# ────────────────────────────────────────────────────────────
# CELL 14 — Era heatmap
# ────────────────────────────────────────────────────────────
"""
era_label_map = {
    'pre_GFC_bull':    '2004–07 Pre-GFC Bull',
    'crisis_recovery': '2008–11 Crisis & Recovery',
    'long_bull':       '2012–19 Long Bull Run',
    'zirp_bubble':     '2020–21 ZIRP Bubble',
    'rate_hike':       '2022–24 Rate Hike Cycle',
}

merged = cluster_rich.copy()
merged['era_label'] = merged['era'].map(era_label_map)

era_order_labels  = list(era_label_map.values())
arch_order_labels = arch_order

cross = pd.crosstab(merged['era_label'], merged['archetype'],
                    normalize='index') * 100
cross = cross.reindex(index=era_order_labels,
                      columns=arch_order_labels, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(cross.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=70)
ax.set_xticks(range(len(arch_order_labels)))
ax.set_xticklabels(arch_order_labels, fontsize=10)
ax.set_yticks(range(len(era_order_labels)))
ax.set_yticklabels(era_order_labels, fontsize=10)
ax.set_title('Distribution of IPO Classifications by Market Era\n'
             '(% of IPOs within each era)',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='% of IPOs in era')
for i in range(len(era_order_labels)):
    for j in range(len(arch_order_labels)):
        val = cross.values[i, j]
        ax.text(j, i, f"{val:.0f}%",
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if val > 45 else 'black')
plt.tight_layout()
plt.savefig('heatmap_final.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Heatmap saved")
"""


# ────────────────────────────────────────────────────────────
# CELL 15 — Random Forest
# ────────────────────────────────────────────────────────────
"""
base = cluster_rich[['ticker','sector','ipo_year',
                      'listing_return','vix_at_ipo',
                      'bhar_1yr','archetype']].dropna().copy()

le = LabelEncoder()
base['sector_enc'] = le.fit_transform(base['sector'].astype(str))

def era_enc(y):
    if 2004 <= y <= 2007: return 0
    if 2008 <= y <= 2011: return 1
    if 2012 <= y <= 2019: return 2
    if 2020 <= y <= 2021: return 3
    if 2022 <= y <= 2024: return 4
    return -1

base['era_enc'] = base['ipo_year'].apply(era_enc)

# Model 1 — listing-day signals only
F1 = ['listing_return','vix_at_ipo','ipo_year','sector_enc','era_enc']
X1 = base[F1].values
y  = base['archetype'].values

# Model 2 — adds bhar_1yr
F2 = F1 + ['bhar_1yr']
X2 = base[F2].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_params = dict(n_estimators=300, max_depth=8, min_samples_leaf=5,
                 random_state=42, class_weight='balanced', n_jobs=-1)

rf1 = RandomForestClassifier(**rf_params)
rf2 = RandomForestClassifier(**rf_params)

scores1 = cross_val_score(rf1, X1, y, cv=cv, scoring='accuracy')
scores2 = cross_val_score(rf2, X2, y, cv=cv, scoring='accuracy')

print(f"V1 (listing-day only): {scores1.mean():.3f} ± {scores1.std():.3f}")
print(f"V2 (+bhar_1yr):        {scores2.mean():.3f} ± {scores2.std():.3f}")

rf1.fit(X1, y)
rf2.fit(X2, y)
"""


# ────────────────────────────────────────────────────────────
# CELL 16 — Feature importance chart
# ────────────────────────────────────────────────────────────
"""
feature_labels = {
    'listing_return': 'Listing Day Return',
    'vix_at_ipo':     'VIX at IPO Date',
    'ipo_year':       'IPO Year',
    'sector_enc':     'Sector',
    'era_enc':        'Market Era',
    'bhar_1yr':       '1-Year BHAR',
}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, rf_m, feats, score, palette, title in zip(
    axes,
    [rf1, rf2],
    [F1, F2],
    [scores1.mean(), scores2.mean()],
    [('#534AB7','#AFA9EC'), ('#1D9E75','#9FE1CB')],
    ['Model 1 — Listing-Day Signals Only',
     'Model 2 — Including 1-Year BHAR']):

    imp = pd.Series(rf_m.feature_importances_,
                    index=[feature_labels.get(f, f) for f in feats]
                    ).sort_values(ascending=True)
    bar_colors = [palette[0] if v > 0.15 else palette[1] for v in imp.values]
    ax.barh(imp.index, imp.values, color=bar_colors)
    ax.set_title(f'{title}\nCV Accuracy: {score*100:.1f}%',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature Importance', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(imp.values):
        ax.text(v+0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.suptitle('Random Forest — Predictors of IPO Long-Run Classification',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('feature_importance_final.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Feature importance saved")
"""


# ────────────────────────────────────────────────────────────
# CELL 17 — F1 summary table
# ────────────────────────────────────────────────────────────
"""
y_pred1 = cross_val_predict(rf1, X1, y, cv=cv)
y_pred2 = cross_val_predict(rf2, X2, y, cv=cv)
r1 = classification_report(y, y_pred1, output_dict=True)
r2 = classification_report(y, y_pred2, output_dict=True)

archetypes = ['Alpha Compounders','Listing-Day Reversals',
              'Gradual Outperformers','Chronic Underperformers']
print("=" * 58)
print("F1 Score by Archetype")
print("=" * 58)
print(f"{'Archetype':<28} {'F1 Day-1':>10} {'F1 +1yr':>10} {'Delta':>10}")
print("-" * 58)
for a in archetypes:
    f1 = r1.get(a, {}).get('f1-score', 0)
    f2 = r2.get(a, {}).get('f1-score', 0)
    print(f"{a:<28} {f1:>10.3f} {f2:>10.3f} {f2-f1:>+10.3f}")
"""


# ────────────────────────────────────────────────────────────
# CELL 18 — THE BUBBLE CHART (LinkedIn visual)
# ────────────────────────────────────────────────────────────
"""
yearly = cluster_rich.groupby('ipo_year').agg(
    mean_bhar_1yr  = ('bhar_1yr',  'mean'),
    count          = ('ticker',    'count')
).reset_index().dropna()

fig, ax = plt.subplots(figsize=(13, 7))

sc = ax.scatter(yearly['ipo_year'], yearly['mean_bhar_1yr'],
                s=yearly['count'] * 0.5,
                c=yearly['mean_bhar_1yr'],
                cmap='RdYlGn', alpha=0.88,
                edgecolors='white', linewidth=1.2, zorder=3)

# Label each year
for _, row in yearly.iterrows():
    ax.annotate(str(int(row['ipo_year'])),
                xy=(row['ipo_year'], row['mean_bhar_1yr']),
                xytext=(0, 11), textcoords='offset points',
                ha='center', fontsize=10, color='#333333',
                fontweight='500')

# Annotate best and worst
best = yearly.loc[yearly['mean_bhar_1yr'].idxmax()]
worst = yearly.loc[yearly['mean_bhar_1yr'].idxmin()]
ax.annotate(f"{int(best['ipo_year'])}: best cohort\n+{best['mean_bhar_1yr']:.0f}% vs S&P 500",
            xy=(best['ipo_year'], best['mean_bhar_1yr']),
            xytext=(best['ipo_year'] - 3, best['mean_bhar_1yr'] + 3),
            fontsize=8.5, color='#1D6E54',
            arrowprops=dict(arrowstyle='->', color='#1D6E54', lw=1))
ax.annotate(f"{int(worst['ipo_year'])}: {int(worst['count'])} IPOs\nWorst cohort {worst['mean_bhar_1yr']:.0f}%",
            xy=(worst['ipo_year'], worst['mean_bhar_1yr']),
            xytext=(worst['ipo_year'] - 4, worst['mean_bhar_1yr'] - 4),
            fontsize=8.5, color='#A32D2D',
            arrowprops=dict(arrowstyle='->', color='#A32D2D', lw=1))

# Zero line
ax.axhline(0, color='grey', lw=1, ls='--', alpha=0.5)

# Clean x-axis — integer years only
ax.set_xticks(yearly['ipo_year'].astype(int).tolist())
ax.set_xticklabels(yearly['ipo_year'].astype(int).tolist(),
                   fontsize=11, rotation=45, ha='right')
ax.set_yticks(range(-40, 25, 10))
ax.set_yticklabels([f'{v}%' for v in range(-40, 25, 10)], fontsize=11)

ax.set_xlabel('IPO Year', fontsize=13)
ax.set_ylabel('Mean 1-Year BHAR vs S&P 500 (%)', fontsize=13)
ax.set_title('Mean 1-Year BHAR by IPO Year\n'
             'Bubble size = number of IPOs that year',
             fontsize=14, fontweight='bold', pad=15)

# Bubble size legend
for n, label in [(50,'~100 IPOs'), (200,'~400 IPOs'), (500,'~1000 IPOs')]:
    ax.scatter([], [], s=n*0.5, c='grey', alpha=0.5,
               edgecolors='white', label=label)
ax.legend(title='IPO Volume', fontsize=9,
          title_fontsize=9, loc='lower left', framealpha=0.9)

ax.grid(True, alpha=0.15)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Mean 1-yr BHAR (%)', fontsize=11)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('bubble_chart_clean.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Bubble chart saved — ready for LinkedIn")
"""


# ────────────────────────────────────────────────────────────
# CELL 19 — Save all datasets
# ────────────────────────────────────────────────────────────
"""
master.to_csv('ipo_master.csv', index=False)
cluster_rich.to_csv('ipo_clustered.csv', index=False)
print("✓ Saved: ipo_master.csv")
print("✓ Saved: ipo_clustered.csv")

from google.colab import files
files.download('ipo_master.csv')
files.download('ipo_clustered.csv')
"""
