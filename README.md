# Breast Cancer Diagnostic Clustering

Unsupervised K-Means clustering of the Wisconsin breast cancer diagnostic dataset (569 tumors, 30 features), comparing raw vs. PCA-reduced feature spaces and validating the resulting clusters against the actual `diagnosis` label.

**Read [Revision Context](#revision-context-the-original-notebook-clustered-a-different-dataset) before assuming this file has always described this dataset.**

## Contents

```
.
├── Cancer_Clustering_corrected.ipynb   # main analysis notebook
├── data/
│   └── cancer.csv                      # source dataset (569 rows, 32 columns)
└── README.md
```

## Dataset

The Wisconsin breast cancer diagnostic dataset — 569 tumors, 32 columns, no missing values, no duplicate rows:

| Column | Type | Notes |
|---|---|---|
| id | identifier | dropped before modeling |
| diagnosis | categorical | M (malignant) / B (benign) — **held out of clustering features, used only for post-hoc validation** |
| radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, ... (30 total) | numeric | mean, standard error, and "worst" value of 10 measured cell-nucleus characteristics |

Class balance: 357 benign (62.7%), 212 malignant (37.3%).

## Revision context: the original notebook clustered a different dataset

An earlier draft of this notebook loaded this cancer data, looked at it for a handful of cells (`.head()`, `.info()`, `.describe()`, null/duplicate checks), and then **silently overwrote the working dataframe with a completely unrelated dataset** — the UCI "Wholesale Customers" dataset (440 rows of retail spending by `Channel` and `Region`). Every subsequent step in that notebook — the elbow method, the final K-Means model, the cluster centers, both silhouette scores — described wholesale-customer spending, not cancer data. The cancer data was never touched again after being loaded.

Separately, decoding that notebook's own stored cluster centers showed its "6 customer segments" were actually just the 6 possible `Channel × Region` combinations, recovered with near-perfect purity — the same one-hot-dummy-dominates-Euclidean-distance failure mode documented in a companion customer-segmentation project in this series. That issue doesn't apply here: `diagnosis` (the only categorical column) is excluded from the clustering features entirely.

This notebook clusters the actual cancer data, end to end, and — unlike the prior two projects in this series — the strong result that comes out of it is real, not an artifact. See below.

## Methodology

1. **Data Loading & Cleaning** — load, check nulls/duplicates, drop `id`, separate `diagnosis` as a validation-only label (never a clustering input).
2. **Exploratory Data Analysis** — class balance, feature distributions, full correlation heatmap, and an explicit count of highly correlated feature pairs.
3. **Preprocessing** — `StandardScaler` on all 30 features.
4. **Improved Clustering Methodology** — two feature spaces, each with `K` chosen by max silhouette across `K = 2..10`:
   - **Raw standardized**: all 30 features.
   - **PCA-reduced**: enough principal components to retain 95% of variance.
5. **Model Comparison** — K-Means vs. Agglomerative Clustering, on both feature spaces, at `k=2`.
6. **Final Model** — K-Means on PCA-reduced features, `k=2`, evaluated against `diagnosis`.
7. **Biological Sanity Check** — profiles the two clusters on their original (unscaled) feature values to check whether the split makes clinical sense.

### Why PCA at all

The correlation heatmap surfaces real multicollinearity: **15 feature pairs correlate above 0.95**, almost entirely among `radius` / `perimeter` / `area` (mean, standard-error, and "worst" versions of each) — expected, since `perimeter ≈ 2πr` and `area ≈ πr²` are geometric identities, not independent measurements. Feeding all 30 raw features into Euclidean-distance K-Means effectively triple-counts "how big is this tumor" relative to shape-based features like concavity. PCA reduces this redundancy; **10 of the 30 original dimensions retain 95% of the variance.**

Both feature spaces independently select `k=2` as silhouette-optimal — not assumed in advance, and it happens to match the number of real diagnosis classes, which the clustering never saw.

## Model comparison

| Model | Feature space | Silhouette | ARI vs. diagnosis |
|---|---|---|---|
| K-Means | Raw standardized (30 features) | 0.3434 | 0.6536 |
| Agglomerative | Raw standardized (30 features) | 0.3394 | 0.5750 |
| **K-Means** | **PCA-reduced (10 components, 95% var)** | **0.3580** | 0.6707 |
| Agglomerative | PCA-reduced (10 components, 95% var) | 0.2960 | **0.7019** |

*ARI = Adjusted Rand Index against the true `diagnosis` label — 0 is random, 1 is perfect agreement. Computed only for validation; `diagnosis` was never a clustering input.*

No combination wins on both metrics. **K-Means on PCA-reduced features is the final model**, selected on silhouette — the metric available in any unsupervised setting where a ground-truth label like `diagnosis` might not exist. If matching the known clinical label as closely as possible is the actual goal, Agglomerative on PCA-reduced features has the higher ARI instead; that's a legitimate alternative, not a worse model — it's optimizing agreement with an external label rather than intrinsic cluster geometry.

## Results

Final model (K-Means, PCA-reduced, `k=2`):

| | Predicted Cluster 0 (n=380) | Predicted Cluster 1 (n=189) |
|---|---|---|
| **Actual Benign** (357) | 343 | 14 |
| **Actual Malignant** (212) | 37 | 175 |

Cluster 0 is 90.3% benign; Cluster 1 is 92.6% malignant. **Silhouette: 0.358. ARI vs. diagnosis: 0.6707.**

The biological sanity check (Section 7) shows *why*: Cluster 1 (malignant-leaning) averages ~40% larger radius/perimeter and 3–4x higher concavity and concave-points scores than Cluster 0 — bigger, more irregularly-shaped tumors. Larger size and boundary irregularity are established malignancy indicators in oncology. The clustering was never told the diagnosis and was never told which features matter for it — it re-derived a real, clinically meaningful distinction from the shape of the data alone. Unlike the segmentation and churn projects in this series, where a similarly good-looking metric turned out to be fake (leaked label) or an artifact (a recovered categorical confound), this result held up under the same scrutiny.

## Limitations

- **This is not a diagnostic tool.** ~9–10% of cases in each cluster carry the "wrong" diagnosis. Clustering optimizes geometric separation, not diagnostic accuracy — some malignant tumors resemble benign ones on these measurements, and vice versa. An ARI of 0.65–0.70 is a strong unsupervised result, not a clinical-grade classifier.
- **`k=2` matching the true class count is a result, not an assumption** — it fell out of the silhouette sweep independently in both feature spaces — but this dataset happens to have exactly two real classes to check against. On data without a known label, there's no equivalent sanity check available.
- **The 95%-variance PCA threshold is a reasonable default, not the only valid choice.** A different threshold, or a scree-plot elbow on the components themselves, could shift the downstream comparison.
- **Only two algorithms and two feature spaces were compared.** DBSCAN, Gaussian Mixture Models, and other dimensionality reductions (e.g. UMAP) were not explored.
- **K-Means vs. Agglomerative is a genuine, unresolved tradeoff** (see Model comparison) — pick based on whether label-free cluster quality (silhouette) or agreement with a known label (ARI) matters more for the use case.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Verified working with:

| Package | Version |
|---|---|
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |

(Older pandas 2.x / earlier scikit-learn versions should also work — nothing here depends on pandas 3.x-specific behavior.)

## Running the notebook

```bash
jupyter notebook Cancer_Clustering_corrected.ipynb
```

Run **Kernel → Restart & Run All**. The notebook is verified to execute top-to-bottom with zero errors and sequential execution counts (1 → 17) against the dataset in `data/`.

## Revision notes

This notebook was rebuilt from an earlier draft that never actually analyzed the cancer data:

- Removed the entire wholesale-customer detour — the earlier notebook silently swapped in an unrelated dataset partway through (see Revision Context) and never returned to the cancer data.
- Fixed the hardcoded, machine-specific file path (`C:\Users\User\Downloads\cancer.csv`, which also didn't match the actual `.xlsx` source file) in favor of a relative `data/` path.
- Removed a broken, argument-less `sns.histplot()` call that rendered an empty plot.
- Added proper exploratory analysis, including an explicit multicollinearity check that motivates the PCA comparison.
- Replaced an eyeballed cluster count with `K` selected by cross-validated-style silhouette sweep in two feature spaces.
- Added a full model comparison (K-Means vs. Agglomerative × raw vs. PCA features) with both silhouette and Adjusted Rand Index against the real `diagnosis` label — the validation step the original notebook had no equivalent of, since it never used a dataset with a checkable ground truth in the first place.
- Added a biological sanity-check section profiling clusters on their original feature scale, to confirm the statistical result reflects a real, interpretable distinction and not another artifact.
