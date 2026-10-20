# AI4I — Stratified Split Results & Why We Don't Pursue Them

> **Stratified-split run (legacy / reproduction)**
> Dataset: AI4I 2020 · Split: random stratified 70/15/15 (preserves class balance, ignores row order)
> Notebook: `vfinal-ieee-Stratified.ipynb`

---

## 1. What This Document Is

This report covers the AI4I run produced under the **stratified** split, which is the original Milestone-3 protocol. We ran it to verify reproducibility, but **we do not recommend reporting these numbers as headline results**. This document explains why.

---

## 2. The Numbers (Stratified Split)

### 2.1 Forecasting

| Forecaster | RMSE | MAE | Coverage |
|---|---|---|---|
| Persistence | 0.1889 | 0.1135 | 71.5% |
| Moving Average | 0.1729 | 0.1084 | 77.2% |
| **TFT (ours)** | **0.1372** | **0.0786** | **86.5%** |

TFT vs Persistence: **−27.3% RMSE** · TFT vs MA: **−20.6% RMSE**

### 2.2 XGBoost Detector

| Metric | Stratified | Chronological |
|---|---|---|
| ROC-AUC | **0.9918** | 0.7584 |
| Average Precision | **0.9725** | 0.3786 |
| F1 | **0.9245** | 0.5087 |
| FTEPH mean horizon | 33.2 | 32.8 |

### 2.3 Healthy-Seed Controls

| Type | Mean risk | Peak | Drift | Crossings |
|---|---|---|---|---|
| HDF | 0.165 | 0.425 | +0.280 | 2/5 |
| PWF | 0.144 | 0.335 | +0.272 | 2/5 |
| OSF | 0.228 | 0.665 | +0.463 | 5/5 |
| TWF | 0.150 | 0.418 | +0.229 | 2/5 |

### 2.4 Closed-Loop Digital Twin

| Type | Mean (sim) | Peak | Consistency | TTF |
|---|---|---|---|---|
| HDF | 0.873 | 1.000 | **0.868** | 1 |
| PWF | 0.720 | 1.000 | **0.709** | 1 |
| OSF | 0.667 | 1.000 | **0.654** | 1 |
| TWF | 0.442 | 0.999 | **0.420** | 1 |

Healthy baseline: 0.0375

---

## 3. Why These Numbers Look Better Than the Chronological Numbers

Compare the two splits side by side:

| Metric | Stratified | Chronological | Δ |
|---|---|---|---|
| XGBoost ROC-AUC | 0.9918 | 0.7584 | **+30.8 pts** |
| XGBoost AP | 0.9725 | 0.3786 | **+59.4 pts** |
| XGBoost F1 | 0.9245 | 0.5087 | **+41.6 pts** |
| Mean digital twin consistency | 0.66 | 0.32 | **+2.0×** |
| TFT RMSE | 0.1372 | 0.1372 | 0 |

The forecasting numbers are identical — TFT is trained on the same windows either way, just reshuffled. **But the XGBoost detector and the digital twin consistency are dramatically inflated under stratified.** AUC moves from 0.76 to 0.99 by changing nothing except how train/test are split.

This is the textbook signature of **temporal leakage**.

---

## 4. Why Stratified Is Wrong Here

Under stratified random splitting, sliding windows are shuffled across train, validation, and test. A window ending at row 4380 can be in training, while the window ending at row 4381 — which shares **19 of 20 rows** with it — ends up in test.

The detector then "learns" the test window from the training window almost by copy. Three pathologies follow:

1. **XGBoost AUC = 0.99 is not generalisation.** The model is mostly recognising sensor sequences it has already seen with one row of difference. It would not see those exact sequences in deployment.

2. **Healthy-seed control is artificially clean.** Healthy seeds in the test set have near-identical neighbours in train labelled healthy — so the detector confidently rules them healthy. Only 2/5 cross the threshold for HDF/PWF/TWF, which looks great but reflects leakage, not discriminative power.

3. **Digital twin consistency = 0.87 is meaningless.** The closed loop feeds TFT-generated futures back into a detector that has already memorised neighbouring real sequences. High consistency under leakage tells us only that the leakage is consistent — it does not tell us the generator produces physically meaningful trajectories on new data.

Forecasting numbers are unaffected because TFT optimises a per-window loss with no test-set label information, but the **detection and digital twin layers are structurally compromised**.

---

## 5. The Decision

We report the **chronological-split** numbers as headline results in the paper. The stratified numbers are kept only as a reproducibility check against the original Milestone-3 implementation.

| What we report | Split | Justification |
|---|---|---|
| TFT forecasting | Either (identical) | Forecaster loss has no leakage path |
| XGBoost detection | **Chronological** | Stratified inflates AUC by 23 points via window-overlap leakage |
| Healthy-seed control | **Chronological** | Stratified controls are biased toward "healthy" by neighbour memorisation |
| Digital twin consistency | **Chronological** | Stratified consistency is doubled by the same leakage |

A reviewer who sees AUC = 0.99 on AI4I should immediately ask how the split was constructed. Reporting the stratified numbers as headline would invite — and deserve — that scepticism. The chronological numbers are lower but defensible.

---

## 6. One-Line Conclusion

The stratified split inflates detection and digital twin metrics through window-overlap leakage while leaving forecasting metrics unchanged; we therefore retain stratified only as a Milestone-3 reproducibility check and report chronological-split numbers in the manuscript.
