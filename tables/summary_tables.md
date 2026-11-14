# Publication Summary Tables: Count-Min Sketch Empirical Evaluation

This document provides visual scorecard and heatmap tables summarizing the core empirical findings of the project ($T = 5{,}000$ independent trials per condition).

---

## Table 1: Frequency Stratum Scorecard Matrix ($\varepsilon = 0.003, w = 907, d = 3$)

Evaluated on the Real Dostoevsky Corpus ($N = 870{,}124$ tokens, $V = 18{,}428$ vocabulary words, 20 representative items per stratum).

| Frequency Stratum | Rank Range ($r$) | True Frequency Range $c(x)$ | Mean Absolute Overestimation ($\bar{\Delta}$) | Median Relative Error $\frac{\hat{c}-c}{c}$ (%) | Empirical Failure Rate $\hat{P}_{\mathrm{fail}}$ | Markov Bound ($\delta$) | Theoretical Slack Multiplier | Operational Usability Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Head (Top 20)** | $1 \le r \le 20$ | $6{,}844 \text{ – } 35{,}319$ | $230 \pm 18$ counts | **$1.22\%$** | $0.00028$ ($\le 0.06\%$) | $0.0500$ ($5.0\%$) | **$178\times$ safer** | 🟢 **Safe & Highly Accurate** (Ideal for exact frequency queries & heavy-hitter reporting) |
| **Middle (Median)** | $200 \le r \le 500$ | $24 \text{ – } 58$ | $233 \pm 22$ counts | **$61.94\%$** | $0.00013$ ($\le 0.06\%$) | $0.0500$ ($5.0\%$) | **$384\times$ safer** | 🟡 **Boundary Zone** (Additive error matches true frequency; ranks preserved, but counts distorted) |
| **Tail (Bottom Decile)**| $r > 16{,}500$ | $1 \text{ – } 2$ | $235 \pm 24$ counts | **$23{,}500\%$** | $0.00018$ ($\le 0.06\%$) | $0.0500$ ($5.0\%$) | **$277\times$ safer** | 🔴 **Catastrophic Failure** ($0.0\%$ pass rate; collision noise completely overwhelms point estimates) |

> **Key Takeaway:** The absolute collision noise is nearly constant ($\approx 230$ counts) across all vocabulary strata. Consequently, relative error is entirely determined by true frequency ($c(x)$), dividing the vocabulary into a reliable head and an unusable tail.

---

## Table 2: Zipf Skew Sensitivity & Reliability Matrix ($\varepsilon = 0.003, w = 907, d = 3$)

Empirical transition metrics for achieving $\le 10\%$ relative error in $\ge 95\%$ of trials across synthetic Zipf streams ($s \in \{0.6, 0.8, 1.0, 1.2, 1.5\}$) and the Real Corpus.

| Dataset | Zipf Exponent ($s$) | Frequency Threshold $f^*_{\mathrm{binned}}$ | Frequency Threshold $f^*_{\mathrm{strict}}$ | Rank of $f^*$ | Peak Pass Rate $\max(\bar{q})$ | Head Passing % ($r \le 500$) | Tail Passing % ($r > 500$) | Convergence Status & Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `zipf_0.6` | $s = 0.6$ (Heavy Tail) | **NaN** | **NaN** | — | $64.54\%$ | **$0.0\%$** | **$0.0\%$** | 🔴 **Non-Convergent:** Count mass smeared across tail; no word achieves $95\%$ reliability |
| `zipf_0.8` | $s = 0.8$ (Moderate-Low) | $7{,}142$ counts | $7{,}692$ counts | 5 | $100.0\%$ | **$1.0\%$** | **$0.0\%$** | 🟡 **High Barrier:** Only the top 5 dominant words reach usable accuracy |
| `zipf_1.0` | $s = 1.0$ (Canonical Zipf) | $4{,}074$ counts | $4{,}428$ counts | 21 | $100.0\%$ | **$3.8\%$** | **$0.0\%$** | 🟢 **Moderate Coverage:** Top 21 words meet the $10\%$ error threshold |
| **Real Corpus** | *Empirical Text* | $3{,}068$ counts | $3{,}271$ counts | 39 | $100.0\%$ | **$7.4\%$** | **$0.0\%$** | 🟢 **Empirical Alignment:** Aligns closely with $s \in [1.0, 1.2]$ linguistic models |
| `zipf_1.2` | $s = 1.2$ (Moderate-High) | $2{,}113$ counts | $2{,}006$ counts | 40 | $100.0\%$ | **$8.4\%$** | **$0.0\%$** | 🟢 **Broadening Coverage:** Noise floor drops, admitting ranks $1 \le r \le 40$ |
| `zipf_1.5` | $s = 1.5$ (Hyper-Skewed) | $379$ counts | $426$ counts | 91 | $100.0\%$ | **$17.2\%$** | **$0.0\%$** | 🟢 **Massive Usable Head:** Dominant head words suppress tail collision noise $18.8\times$ |

> **Key Takeaway:** As skew increases from $0.8$ to $1.5$, the required frequency threshold $f^*$ drops by **$18.8\times$** (from $7{,}142$ to $379$), expanding usable head vocabulary from $1.0\%$ to $17.2\%$. Across all distributions, **$0.0\%$ of tail items ($r > 500$) ever achieved reliable relative accuracy**.

---

## Table 3: Depth Ablation Multiplier Matrix ($d=1$ vs. $d=3$)

Quantifying the exact mathematical benefit of multi-row hashing across width settings ($\varepsilon$) and datasets ($T = 5{,}000$ independent trials).

| Dataset | Target Error ($\varepsilon$) | Width ($w$) | Frequency Class | Median $\hat{P}_{\mathrm{fail}}$ ($d=1$) | Median $\hat{P}_{\mathrm{fail}}$ ($d=3$) | Single-Row Noise ($\frac{\bar{\Delta}_{d=1}}{N}$) | Multi-Row Noise ($\frac{\bar{\Delta}_{d=3}}{N}$) | **Error Reduction Multiplier** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real Corpus** | $0.001$ | $2{,}719$ | Head ($r \le 20$) | $3.98\%$ | $0.02\%$ | $0.000351$ | $0.000040$ | **$8.87\times$ Cleaner** |
| | $0.001$ | $2{,}719$ | Tail ($r > 500$) | $5.69\%$ | $0.02\%$ | $0.000380$ | $0.000050$ | **$7.58\times$ Cleaner** |
| | $0.003$ | $907$ | Head ($r \le 20$) | $4.56\%$ | $0.02\%$ | $0.001067$ | $0.000230$ | **$4.63\times$ Cleaner** |
| | $0.003$ | $907$ | Middle ($200 \le r \le 500$) | $6.36\%$ | **$0.00\%$** | $0.001119$ | $0.000266$ | **$4.21\times$ Cleaner** |
| | $0.003$ | $907$ | Tail ($r > 500$) | $6.09\%$ | $0.02\%$ | $0.001094$ | $0.000266$ | **$4.12\times$ Cleaner** |
| | $0.010$ | $272$ | Head ($r \le 20$) | $4.92\%$ | **$0.00\%$** | $0.003511$ | $0.001260$ | **$2.79\times$ Cleaner** |
| | $0.010$ | $272$ | Tail ($r > 500$) | $6.48\%$ | $0.03\%$ | $0.003686$ | $0.001395$ | **$2.64\times$ Cleaner** |
| **Synthetic Zipf ($s=1.0$)** | $0.001$ | $2{,}719$ | Head ($r \le 20$) | $3.30\%$ | $0.02\%$ | $0.000339$ | $0.000076$ | **$4.43\times$ Cleaner** |
| | $0.001$ | $2{,}719$ | Tail ($r > 500$) | $4.33\%$ | **$0.00\%$** | $0.000370$ | $0.000086$ | **$4.31\times$ Cleaner** |
| | $0.003$ | $907$ | Head ($r \le 20$) | $3.47\%$ | **$0.00\%$** | $0.001051$ | $0.000346$ | **$3.04\times$ Cleaner** |
| | $0.003$ | $907$ | Middle ($200 \le r \le 500$) | $4.54\%$ | **$0.00\%$** | $0.001111$ | $0.000374$ | **$2.97\times$ Cleaner** |
| | $0.003$ | $907$ | Tail ($r > 500$) | $4.43\%$ | **$0.00\%$** | $0.001126$ | $0.000374$ | **$3.01\times$ Cleaner** |
| | $0.010$ | $272$ | Head ($r \le 20$) | $3.59\%$ | **$0.00\%$** | $0.003516$ | $0.001570$ | **$2.24\times$ Cleaner** |
| | $0.010$ | $272$ | Tail ($r > 500$) | $4.51\%$ | $0.01\%$ | $0.003701$ | $0.001663$ | **$2.23\times$ Cleaner** |

> **Key Takeaway:** Multi-row hashing ($d=3$) suppresses failure probabilities from $3\%\text{--}7\%$ down to $\le 0.06\%$ ($53.6\%$ of items experienced literally zero failures in 5,000 trials). Simultaneously, taking the row minimum reduces class-median estimation error by **$2.23\times$ to $8.87\times$**.
